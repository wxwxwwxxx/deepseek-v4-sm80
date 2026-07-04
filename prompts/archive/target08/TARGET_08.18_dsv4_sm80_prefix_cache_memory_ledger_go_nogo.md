# TARGET 08.18: DSV4 Prefix Cache Memory Ledger And 08.20 Go/No-Go

## Status

Planned after TARGET 08.10.

This is a short analysis target.  It should not implement a new cache component
or allocator.

## Goal

Decide whether TARGET 08.20, the SGLang-style SWA/component retention target,
is worth doing.

The phase-1 DSV4 prefix cache keeps full-token pages as the canonical owner.
That is correct and simple, but it may retain more memory than SGLang's
independent SWA/component/tombstone design.  This target computes the memory and
capacity tradeoff before starting a risky memory-model rewrite.

## Inputs

Read:

- `performance_milestones/target08_radix_prefix_dsv4/README.md`
- TARGET 08.05 result README;
- TARGET 08.06 result README;
- TARGET 08.07 result README;
- TARGET 08.10 result README;
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`

## Fixed Memory Inputs From TARGET 08.06 And 08.07

Carry these costs into the ledger unless a later target reruns the same
configuration and proves they changed:

| Item | Value | Source / interpretation |
| --- | ---: | --- |
| CUDA graph private-pool capture cost | `~19.04 GiB/rank` | TARGET 08.06/08.07 full bucket `[1,2,4,8,16]`; this is a real first-graph dominated private-pool capacity cost. |
| First captured graph cost | `~18.83 GiB/rank` | TARGET 08.06/08.07 single bucket `[16]`; later buckets add only tens of MiB because graph pool reuse works. |
| Promoted tested BF16 cache persistent baseline | `1.588 GiB/rank` | TARGET 08.07 model prepare; visible before graph capture, not the cause of graph private-pool delta. |
| Fixed KV/page capacity at `--num-pages 128`, page size `256` | `~2.320 GiB/rank` | TARGET 08.06-compatible run shape; use as the capped-page baseline for prefix-cache serving tests. |

TARGET 08.07 specifically ruled out the promoted BF16 projection/shared-expert
cache paths as a material owner of the `~19 GiB/rank` graph delta:

- disabling all tested BF16 caches removed the `1.588 GiB/rank` persistent
  baseline from `model_prepare_report`;
- the `[16]` graph delta changed only from `18.828 GiB/rank` to
  `18.885 GiB/rank`;
- individual cache-owner A/B rows moved graph delta by at most
  `~0.057 GiB/rank`;
- graph replay stayed active.

Therefore this target should treat BF16 caches and graph private-pool memory as
separate capacity ledger lines.  Do not spend this target trying to identify the
internal CUDA graph node/workspace owner unless the capacity ledger shows that
the graph pool cost blocks prefix-cache promotion.

## Required Analysis

Build a memory ledger for prefix retention under several workloads:

- short shared prefix;
- 1024-token prefix;
- 4096-token prefix;
- multiple distinct prefixes;
- sustained serving workload from TARGET 08.10;
- eviction-pressure workload from TARGET 08.10.

For each case, estimate or measure:

- retained full-token pages;
- retained SWA-visible full slots;
- retained C4 slots;
- retained C128 slots;
- retained C4-indexer slots;
- retained compression-state slots;
- retained bytes/rank;
- graph capture private-pool or capture delta bytes from TARGET 08.06/08.07;
- promoted BF16 cache persistent baseline bytes from TARGET 08.07;
- fixed KV/page bytes under the selected `--num-pages` policy;
- remaining free-memory margin after weights, prepared caches, KV pages, graph
  private pool, and retained prefix pages;
- equivalent KV pages;
- equivalent KV tokens;
- equivalent number of 4096-token prompts;
- equivalent number of 4096+1024 requests;
- impact on max context/concurrency under fixed page count.

Then combine prefix-retention memory with graph-capture memory, because both
reduce usable serving capacity under fixed device memory.  After that, estimate
the theoretical upper bound of SGLang-style independent
SWA/component retention:

- how many full pages could be freed while retaining only needed SWA window or
  compressed components;
- how much memory could be saved;
- how many extra KV tokens/pages that memory buys;
- whether the savings are likely to improve performance, capacity, or only
  bookkeeping elegance.

## Deliverables

Create:

```text
performance_milestones/target08_prefix_cache_memory_ledger/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- formulas and assumptions;
- measured data from TARGET 08.10 where available;
- memory ledger tables;
- SGLang-style theoretical savings estimate;
- engineering risk assessment;
- explicit TARGET 08.20 go/no-go decision.

## Go Criteria For TARGET 08.20

Recommend TARGET 08.20 only if at least one is true:

- phase-1 full-page retention consumes a large fraction of KV capacity under
  realistic serving workloads;
- retained prefixes materially reduce max useful context or concurrency;
- eviction pressure causes latency instability that a component-level retention
  design would plausibly fix;
- SGLang-style retention would recover enough pages/tokens to justify the
  correctness and allocator complexity.

As a rough starting threshold, investigate 08.20 if retained prefix memory is
regularly above `20%-30%` of the fixed KV pool or if it removes multiple
4096+1024 request equivalents of capacity.

## No-Go Criteria

Skip TARGET 08.20 if:

- memory savings are small in realistic workloads;
- eviction pressure is already stable;
- the added component ownership complexity is larger than the capacity gain;
- prefix-cache promotion can proceed as controlled opt-in without independent
  SWA/component retention.

## Stop Rules

Stop and report blocked if:

- existing metrics are insufficient to estimate retained memory and adding the
  required measurements would become a separate implementation target;
- SGLang's component model cannot be mapped to mini's current DSV4 pool without
  changing correctness assumptions;
- capacity calculations depend on automatic KV sizing that is still graph-OOM
  unsafe.

## Non-Goals

- Implementing independent SWA/component retention.
- Changing prefix-cache promotion status.
- Low-precision KV/cache experiments.
- Attention, PyNCCL, or graph bucket tuning.

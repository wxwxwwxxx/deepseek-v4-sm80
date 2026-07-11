# DeepSeek V4 SM80 v0.0.0 Release Baseline

## Status

This document is the stable release and performance reference for:

```text
version/tag: v0.0.0
commit:      005f879e73fe9fe7a1e74f3adedf1c8eeceed41b
date:        2026-07-11 CST
```

This tag remains the immutable pre-cleanup performance baseline.  The
downstream release identity selected later is:

```text
distribution:          minisgl
package version:       0.1.0+dsv4.sm80
recommended final tag: v0.1.0-dsv4-sm80
```

The final tag is reserved for the post-cleanup qualified commit.  It does not
replace or move `v0.0.0`; performance comparisons continue to use the baseline
recorded here.

The measurements were collected by TARGET 12.606 while its implementation was
still a target worktree and were then committed and tagged as `v0.0.0`. The
repository was clean and the tag pointed at the commit above when this baseline
was recorded.

Use this document as the default non-MTP comparison point for subsequent
kernel, long-context, scheduler, cache, and serving work. Raw evidence remains
under `performance_milestones/` and is intentionally not tracked by git.

## Supported Platform And Measurement Contract

```text
hardware:       8 x NVIDIA A100-SXM4-80GB, TP8
architecture:   sm80
model:          /models/DeepSeek-V4-Flash
precision:      BF16 release path; MTP disabled
runtime:        CUDA 12.8.2, NCCL 2.26.2-1
communication:  PyNCCL threshold32m
page size:      256
prefill chunk:  8192 tokens
seed:           606
```

Unless explicitly stated otherwise, performance rows are closed, single-wave
offline workloads. All requests were submitted before execution and a row was
accepted only when all requested sequences fit simultaneously and the trace
showed 1,023 consecutive decode steps at the requested active M. No pending,
staggered-admission, or multi-wave timing is reported as a performance result.

## Release Recipes

Ordinary use requires no DSV4 tuning flags:

```python
from minisgl import LLM

llm = LLM("/models/DeepSeek-V4-Flash")
```

The no-env DSV4 A100/sm80 path resolves to `dsv4_sm80_balanced`.

| Public recipe | Max running requests | CUDA graph max M | Intended use |
| --- | ---: | ---: | --- |
| `dsv4_sm80_low_m64` | 256 | 64 | low-M or KV-capacity-sensitive serving |
| `dsv4_sm80_mid_m128` | 256 | 128 | M<=128 capacity/throughput compromise |
| `dsv4_sm80_balanced` | 256 | 256 | ordinary throughput-oriented default |
| `dsv4_sm80_long_context_512k` | 4 | 4 | low-concurrency 512 Ki-token capability |
| `dsv4_sm80_1m_smoke` | 1 | 1 | 1 Mi-token capability smoke only |

Named recipes are selected with `dsv4_sm80_recipe=` or server option
`--dsv4-sm80-recipe`. Explicit request-capacity, graph, max-sequence, KV,
memory-ratio, and chunk-budget settings remain authoritative. A legal active M
above the selected graph maximum runs eagerly and is observable in telemetry.

No req512/high-concurrency release recipe exists in v0.0.0. Fixed SWA/request
state consumes too much per-rank memory for a useful KV budget at that request
capacity.

## Provisional DGX A100 Performance Card

Every row uses output length 1,024 per request. Latencies are p50/p95 seconds.

| Recipe / active M / prompt | Requests/s | Output tok/s | Prefill tok/s | Decode tok/s | TTFT | TPOT | Replay/eager |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| graph64 / 4 / 16K | 0.0890 | 91.11 | 3,345.76 | 187.59 | 13.771 / 20.487 | 0.03048 / 0.03705 | 1,023 / 0 |
| graph64 / 16 / 4K | 0.3154 | 322.94 | 3,806.03 | 558.60 | 11.330 / 18.940 | 0.03841 / 0.04630 | 1,023 / 0 |
| graph64 / 64 / 4K | 0.4895 | 501.21 | 4,077.24 | 1,191.26 | 37.221 / 68.473 | 0.09082 / 0.12143 | 1,023 / 0 |
| graph128 / 128 / 1K | 1.0356 | 1,060.42 | 5,236.73 | 1,512.68 | 16.044 / 29.107 | 0.10448 / 0.11745 | 1,023 / 0 |
| graph256 / 256 / 1K fresh 1 | 1.1943 | 1,222.97 | 5,395.88 | 1,827.63 | 29.736 / 54.987 | 0.17957 / 0.20425 | 1,023 / 0 |
| graph256 / 256 / 1K fresh 2 | 1.1939 | 1,222.53 | 5,398.30 | 1,826.93 | 29.847 / 55.061 | 0.17955 / 0.20424 | 1,023 / 0 |

The two fresh graph256 processes differed by 0.036% in output throughput and
0.038% in decode-forward time. Physical graph bytes and KV capacity were
identical. This is the repeat-stable balanced-default anchor.

Do not treat the rows above as a pure M-scaling curve: prompt lengths and graph
recipes differ. TARGET 12.61 must use fixed-shape micro/subgraph controls when
attributing backend scaling.

## Graph Memory And KV Capacity

Values are per rank. The conservative 512 MiB graph-reserve margin is shown
separately from the estimator and measured physical graph bytes.

| Graph recipe | Estimated graph bytes | Margin | Actual graph bytes | KV pages | KV tokens | Representative capture time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph64 | 2,017,460,480 | 536,870,912 | 1,501,560,832 | 3,314 | 848,384 | 7.2-7.7 s |
| graph128 | 3,095,396,864 | 536,870,912 | 2,388,656,128 | 3,180 | 814,080 | 10.459 s |
| graph256 | 5,251,269,632 | 536,870,912 | 3,988,783,104 | 2,912 | 745,472 | 18.131-25.320 s |

Graph256 trades 102,912 KV tokens relative to graph64 and 68,608 relative to
graph128 for captured M=129-256 coverage. Cold compilation can increase
graph256 capture to about 45.6 seconds, so startup cost is part of the recipe
tradeoff even though steady-state performance is stable.

## Resident Capacity Envelope

Demand below includes page-rounded prompt plus 1,024 output tokens per request.

| Active M | 1K prompt | 4K prompt | 16K prompt |
| ---: | --- | --- | --- |
| 4 | runnable | runnable | runnable |
| 16 | runnable | runnable | runnable |
| 64 | runnable | runnable | no-go: 1,114,112 required |
| 128 | runnable | runnable | no-go: 2,228,224 required |
| 256 | runnable | no-go: 1,310,720 required | no-go: 4,456,448 required |

The balanced graph256 recipe has 745,472 KV tokens. Capacity-no-go cells were
rejected before loading cell-specific model processes and have no synthetic or
pending-derived performance number.

## Long-Context Capability

| Workload | Required / available tokens | v0.0.0 result |
| --- | ---: | --- |
| one 512 Ki total sequence, req4/graph4 | 524,288 / 1,635,840 | pass: 64 prefill chunks, 7 graph replays |
| aggregate 512 Ki across bs4 | 524,288 / 1,635,840 | planner-runnable; performance not yet measured |
| four independent 512 Ki sequences | 2,097,152 / 1,635,840 | capacity no-go |
| one exact 1 Mi total sequence, req1/graph1 | 1,048,576 / 1,643,264 | pass: 128 prefill chunks, 7 graph replays |

The exact 1 Mi run used `1,048,568` prompt tokens plus eight decode tokens.
512K/1M decode-1K performance is deliberately absent: v0.0.0 establishes
correctness and capacity, while TARGET 12.61 owns long-context backend
attribution and optimization before publication-grade performance is measured.

## Correctness And Release Gate

- Chinese, English, code, arithmetic, and exact-instruction text smoke passed.
- Prefix mixed hit/miss passed with a 40% hit rate and 6,144 saved prefill
  tokens in the promotion workload.
- SWA independent lifecycle, C4/C128 one-surface metadata, live-route MoE
  padding, PyNCCL, sampler, and output gathering remained enabled and healthy.
- Historical 4096x128 bs4 completed with 127 graph replays and zero eager.
- Focused recipe/bucket/benchmark selection passed 99 tests.
- Focused C4/C128/MoE/finite-output gates passed 27 tests.
- Final combined promotion selection passed 216 tests; only upstream
  FlashInfer deprecation warnings remained. Ruff, compileall, and diff checks
  passed.

The release contract does not require cross-bucket BF16 token identity, but it
does require finite outputs, natural-language sanity, valid state/cache
lifecycle, and padded-row/live-route correctness.

## Known Limits And Next Baseline Work

1. Long-context prefill is correct but not yet performance-promoted. Earlier
   evidence attributed about 48% of 512K TTFT to the bounded FP8 indexer.
2. Exact decode M=4/16/64/128/256 showed no anomalous macro scaling that alone
   justifies a large-M kernel rewrite.
3. TARGET 12.61 should re-rank indexer, C4/C128 attention, metadata/cache
   lookup, MoE, and communication at committed-context checkpoints before
   changing dispatch or kernels.
4. After any evidence-backed long-context optimization, regenerate the final
   DGX A100 performance card, including 512K/1M rows that are physically
   meaningful. Do not benchmark capacity-impossible concurrent sequences.

## Evidence

```text
performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md
performance_milestones/target12_cuda_graph_recipe_frontier_selection/README.md
prompts/archive/target12/TARGET_12.606_dsv4_sm80_cuda_graph_recipe_promotion_cleanup.md
prompts/archive/target12/TARGET_12.61_dsv4_sm80_long_context_ttft_owner_attribution.md
```

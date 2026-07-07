# TARGET 09.0: DSV4 SM80 Low-Precision Preflight

## Status

First child target for TARGET 09.

Run this before choosing between INT8 MoE, FP8 KV/cache, INT8 communication, or
projection/cache-boundary fusion.  Do not implement a low-precision runtime path
in this target.

## Goal

Reset the post-TARGET-10 baseline and decide which low-precision lane has enough
evidence to run next.

The target should answer:

- Is MoE compute still large enough to justify INT8 W8A8 research?
- Is cache memory or cache bandwidth large enough to justify FP8 KV/cache?
- Is communication still large enough to justify INT8 communication research?
- Is projection/cache-boundary traffic material enough to justify fusion work?

## Baseline

Use the promoted A100/sm80 DSV4 path:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL default threshold32m
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Mini uses system Python from `/workspace/mini-sglang`.  vLLM, if needed for a
fresh comparison, uses `/workspace/venvs/vllm-dsv4/bin/python` from
`/workspace/vllm-dsv4-docker`.

## Required Work

1. Fresh macro baseline

   Run at least:

   - `historical_4096_128_bs4`;
   - `historical_4096_1024_bs4`;
   - `serving_mixed_112req_wave16`;
   - `prefix_multi_112req_wave16`.

   Record graph replay/eager ratios, token throughput, latency, peak memory, and
   any correctness smoke result.

2. Owner profile reset

   Collect owner timing with enough detail to rank:

   - MoE routed experts and shared experts;
   - attention C4/C128/indexer owners;
   - projection/GEMM owners;
   - cache store/gather/dequant owners;
   - communication owners;
   - metadata/runtime owners.

3. Communication and memory census

   Use `snapshot_communication_stats()` where possible.  Record dtype, shape,
   count, bytes, and labels for TP communication.  Also produce a memory ledger:

   - model weights and cached BF16/Marlin weights;
   - KV/cache pages;
   - prefix-cache component state;
   - CUDA graph memory delta;
   - available context/page headroom.

4. Source census

   Summarize relevant SGLang/vLLM implementation surfaces for:

   - INT8 MoE W8A8 backend candidates;
   - FP8 KV/cache layout and store/gather kernels;
   - INT8 communication or quantized collectives;
   - projection/cache-boundary fusion candidates.

   Mark each observation as runtime-proven, microbench-proven, or source-derived.

5. Decision table

   Recommend one next target:

   - TARGET 09.1 if MoE owner time is material and a plausible SM80 INT8 MoE
     backend exists;
   - TARGET 09.3 if cache memory/capacity or cache bandwidth is the stronger
     reason;
   - TARGET 09.25 if communication remains a top bottleneck and quantized
     communication has a plausible owner shape;
   - TARGET 09.6 only if projection/cache-boundary HBM traffic is unexpectedly
     material;
   - pause TARGET 09 if none of the above has a credible upside.

## Gates

Pass if:

- macro results are repeat-stable enough to compare later targets;
- owner timing identifies the top remaining bottlenecks;
- memory and communication ledgers are concrete, not hand-wavy;
- next-target recommendation is backed by numbers.

Stop if:

- graph replay is broken in the baseline;
- the promoted preset is not actually being used;
- owner timing is too noisy to rank bottlenecks;
- the target starts implementing low-precision features instead of measuring.

## Deliverables

Write results under:

```text
performance_milestones/target09_low_precision_preflight/
```

Include:

- `README.md` with the next-target recommendation;
- raw macro reports and owner timing;
- communication stats JSON/table;
- memory ledger;
- source-census notes;
- exact commands and environment.


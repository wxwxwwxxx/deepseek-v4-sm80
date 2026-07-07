# TARGET 09.45: DSV4 SM80 FP8 Cache ROI And SGLang Lifecycle Alignment

## Status

Run after TARGET 09.4 and before TARGET 09.5.

This is a decision target.  Do not implement full FP8 KV/cache E2E here.  The
goal is to decide whether the FP8 cache route is still worth pursuing after
accounting for:

- future SGLang-aligned SWA lifecycle;
- current mini Route B prefix/component ownership;
- 09.4's measured separated-kernel speed overhead;
- SGLang/vLLM source-aligned FP8 cache layouts.

## Background

TARGET 09.3 found a source-aligned FP8 cache memory upside:

- SWA-only packed MLA saves about `0.576 GiB/rank` at `page_size=256`,
  `num_pages=128`;
- full source-aligned MLA+indexer replacement saves about `0.667 GiB/rank`;
- C4/C128/indexer add only about `0.091 GiB/rank` beyond SWA in the current mini
  allocation model.

TARGET 09.4 proved the SWA packed MLA boundary:

- source-aligned layout passed correctness and graph-safety checks;
- RoPE tail remained BF16;
- selected-row gather/dequant avoided full-cache dequant;
- separated FP8 store + gather/dequant was slower than BF16 by about
  `0.016 ms` per measured bucket.

The open question is whether SWA-only FP8 remains valuable if mini later adopts
SGLang-style independent SWA lifecycle, where SWA can be tombstoned/freed
separately from full/prefix pages.  If SWA storage shrinks, the FP8 SWA memory
win shrinks with it.

## Goal

Produce a concrete go/no-go recommendation for the FP8 cache route:

- proceed to TARGET 09.5 as SWA-only opt-in E2E;
- proceed to TARGET 09.5 with a broader SGLang-aligned MLA/indexer scope;
- defer FP8 cache and first implement SGLang-aligned SWA independent lifecycle;
- stop FP8 cache work for now because memory or speed ROI is insufficient.

## Source References

Mini:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `performance_milestones/target09_fp8_kv_cache_parity_ledger/README.md`
- `performance_milestones/target09_minimal_fp8_kv_cache_slice/README.md`
- TARGET 08 Route B/prefix ownership reports under `performance_milestones/`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/index_buf_accessor.py`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`

## Required Work

1. Current mini memory ledger

   Reproduce the current mini persistent-cache ledger at:

   ```text
   page_size=256
   num_pages=128
   sliding_window=128
   compress_ratios[:43]
   ```

   Include:

   - SWA BF16;
   - C4/C128 BF16 compressed MLA;
   - C4 indexer BF16 and optional additive FP8 side cache;
   - C4/C128/indexer compression state;
   - prefix-retained pages/component state from recent target08/09 scenarios;
   - CUDA graph headroom impact.

2. SGLang-aligned SWA lifecycle model

   Model how memory changes if mini adopts SGLang-style SWA ownership:

   - SWA as a separate component/pool;
   - only sliding-window/tail SWA retained;
   - out-of-window SWA tombstoned/freed independently from full/prefix pages;
   - C4/C128/indexer/state ownership remains independent.

   Use runtime data when available.  If not available, provide scenario-based
   estimates for at least:

   - historical `bs4, 4096/1024`;
   - serving mixed wave16;
   - prefix multi wave16;
   - a higher-concurrency serving scenario if easy to synthesize.

   Report the resulting SWA pool size and the reduced maximum value of SWA-only
   FP8.

3. Source-aligned FP8 layout choices

   Confirm SGLang/vLLM layouts:

   - SWA/C4/C128 MLA: `448` noPE FP8 + `64` RoPE BF16 + UE8M0 scale/pad bytes;
   - C4 indexer: `128` FP8 values + FP32 scale bytes, or optional FP4 indexer;
   - compression state stays BF16/FP32-like;
   - RoPE tail stays BF16.

   Explicitly answer whether C4/C128 noPE should remain BF16.  Source-aligned
   answer should be "no" for the FP8 cache route: keeping noPE BF16 is the
   current mini-like precision path and gives little memory win.

4. Speed ledger

   Use TARGET 09.4 data plus any lightweight probes needed to estimate:

   - cost of separated FP8 SWA store and gather/dequant;
   - likely cost if store is fused into norm/RoPE/store;
   - likely cost if gather/dequant is fused into or placed immediately before
     sparse attention;
   - graph workspace cost per bucket;
   - HBM bytes saved versus additional arithmetic and kernel launch cost.

   Include a "do nothing" BF16 baseline and at least two candidate FP8 designs:

   - separated kernels, known from 09.4;
   - SGLang-aligned fused store + selected-row gather/dequant;
   - optional attention-integrated dequant if source review shows it is realistic.

5. ROI matrix

   Build a decision matrix with memory and speed columns:

   - current mini BF16;
   - current mini + SWA-only FP8;
   - current mini + full source-aligned MLA/indexer FP8;
   - SGLang-aligned SWA lifecycle + BF16;
   - SGLang-aligned SWA lifecycle + SWA-only FP8;
   - SGLang-aligned SWA lifecycle + broader MLA/indexer FP8.

   For each row, report:

   - persistent GiB/rank;
   - graph headroom delta;
   - equivalent pages/tokens;
   - expected latency delta;
   - quality/correctness risk;
   - implementation scope.

6. Recommendation

   Pick exactly one next action:

   - run TARGET 09.5 with SWA-only scope;
   - rewrite TARGET 09.5 scope to broader SGLang-aligned MLA/indexer;
   - write a TARGET 08/09 bridge target for independent SWA lifecycle first;
   - stop/defer FP8 cache and return to another TARGET 09 lane.

## Gates

Proceed to E2E only if:

- capacity value is meaningful after the SWA lifecycle model, not only in the
  current over-retained SWA allocation;
- expected speed cost is acceptable for a capacity mode, or a fused path has a
  plausible route to neutral/positive latency;
- SGLang/vLLM layout parity is preserved;
- prefix/cache/graph correctness risks have concrete guards.

Stop or defer if:

- independent SWA lifecycle removes most of the SWA-only memory value;
- separated or source-aligned kernels would produce a material decode regression
  with no capacity requirement to justify it;
- the only attractive memory win comes from a layout that diverges from
  SGLang/vLLM without a strong reason;
- graph workspace or retained-prefix migration would require a broad rewrite
  before any measurable benefit.

## Deliverables

Write results under:

```text
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/
```

Include:

- `README.md` with one next-action recommendation;
- current mini memory ledger;
- SGLang-aligned SWA lifecycle memory ledger;
- FP8 layout parity notes;
- speed ledger and overhead model;
- ROI decision matrix;
- any scripts used for ledger/probes;
- if continuing, the exact revised TARGET 09.5 scope.

## Non-Goals

- Implementing full FP8 KV/cache E2E.
- Migrating retained prefix pages.
- Changing default cache dtype.
- Quantizing compression state.
- Inventing a non-SGLang/non-vLLM layout before source-aligned ROI is understood.

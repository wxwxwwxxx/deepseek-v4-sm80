# TARGET 09.45 DSV4 SM80 FP8 Cache ROI And SGLang Lifecycle Alignment

Date: 2026-07-05

Scope: decision target only. No full FP8 KV/cache E2E implementation was added.

## Conclusion Summary

Decision: **do not run TARGET 09.5 yet**.

One next action: **write and run an independent SWA lifecycle target first**.

Why:

- Current mini still allocates SWA as a full 43-layer, 128-page BF16 pool. That
  makes SWA-only FP8 look large: **0.576 GiB/rank** saved at 128 SWA pages.
- SGLang-aligned lifecycle changes the denominator. With page-aligned
  `sliding_window=128` and `page_size=256`, long-prefix/prefix-wave scenarios
  need about **4 to 16 SWA tail pages**, not 128 historical SWA pages. In that
  model SWA-only FP8 drops to **0.018 to 0.072 GiB/rank** for the long/prefix
  scenarios that motivated prefix retention.
- Lifecycle + BF16 gives a larger and lower-precision-risk capacity win than
  SWA-only FP8: the canonical wave16 estimate is **1.127 GiB/rank**, a
  **+1.176 GiB/rank persistent headroom delta** versus current mini's promoted
  128-page formula.
- TARGET 09.4 proved layout/correctness/graph safety for the isolated FP8
  boundary, but the measured separated kernels are slower by **+0.016 to
  +0.018 ms per cache boundary**. That is not a production shape without fused
  store and selected-row/attention-integrated dequant.
- SGLang/vLLM parity says the source-aligned FP8 route quantizes noPE for
  SWA/C4/C128 MLA. Keeping C4/C128 noPE BF16 is basically the current mini
  precision route and leaves little additional memory upside.

Therefore 09.5 is deferred until mini has an SGLang-aligned SWA component/pool
lifecycle, tombstone/free behavior, and runtime counters that prove how many
SWA tail pages remain in the real workloads.

Evidence mix:

- Runtime-proven: TARGET 09.0 macro memory/graph/prefix data and TARGET 09.4
  slice correctness, microbench, and graph replay.
- Source-derived: mini pool ownership, SGLang/vLLM FP8 layout, and SGLang SWA
  lifecycle sources.
- Estimated: independent SWA lifecycle memory because mini does not runtime
  implement/prove it yet.

## Current Mini Memory Ledger

Inputs:

```text
page_size=256
num_pages=128
sliding_window=128
compress_ratios[:43] => 21 C4 layers, 20 C128 layers, 2 normal layers
head_dim=512
index_head_dim=128
```

Mini source evidence:

- `python/minisgl/kvcache/deepseek_v4_pool.py:311` allocates SWA as
  `(num_layers, num_pages, page_size, head_dim)`.
- `python/minisgl/kvcache/deepseek_v4_pool.py:313` to `:327` allocate C4,
  C128, and C4 indexer BF16 component pools.
- `python/minisgl/kvcache/deepseek_v4_pool.py:328` to `:340` allocate the
  existing additive paged FP8 indexer side cache when
  `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE` is active.
- `python/minisgl/kvcache/deepseek_v4_pool.py:421` to `:452` allocate C4,
  C128, and indexer compression state pools.
- `python/minisgl/kvcache/deepseek_v4_pool.py:711` to `:800` define the
  retained-prefix component/state estimator used by the runtime reports.

Component formula at the requested 128 pages:

| component | bytes/page | MiB/page | GiB at 128 pages | evidence |
| --- | --- | --- | --- | --- |
| swa_bf16 | 11,272,192 | 10.75 | 1.344 | source-derived |
| c4_bf16 | 1,376,256 | 1.31 | 0.164 | source-derived |
| c128_bf16 | 40,960 | 0.04 | 0.005 | source-derived |
| c4_indexer_bf16 | 344,064 | 0.33 | 0.041 | source-derived |
| c4_indexer_fp8_side_cache | 177,408 | 0.17 | 0.021 | runtime-active in promoted bundle, source-derived size |
| c4_state_bf16 | 688,128 | 0.66 | 0.082 | source-derived |
| c4_indexer_state_bf16 | 172,032 | 0.16 | 0.021 | source-derived |
| c128_state_bf16 | 5,242,880 | 5.00 | 0.625 | source-derived |
| total_with_existing_fp8_side | 19,313,920 | 18.42 | 2.302 | source-derived formula; matches promoted indexer FP8 side mode |
| total_without_fp8_side_reference | 19,136,512 | 18.25 | 2.281 | source-derived formula |

TARGET 09.0 runtime report: `2,491,495,680 B / 2.320 GiB` per rank. That is
exactly **129 pages** at the promoted formula with additive indexer FP8 side
cache. This README uses the user-requested **128-page** formula for ROI and
keeps the 129-page runtime pool as separate runtime-proven context.

Prefix-retained component state from TARGET 09.0:

| scenario | retained pages | retained GiB | SWA GiB | non-SWA component/state GiB | available component pages | live full pages | evictions | saved prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_4096_128_bs4 | 64 | 1.151 | 0.672 | 0.479 | 65 | 4 | 0 | 0 |
| historical_4096_1024_bs4 | 124 | 2.230 | 1.302 | 0.929 | 5 | 11 | 1 | 0 |
| serving_mixed_112req_wave16 | 106 | 1.907 | 1.113 | 0.794 | 23 | 23 | 3 | 0 |
| prefix_multi_112req_wave16 | 87 | 1.565 | 0.913 | 0.652 | 42 | 28 | 4 | 49152 |

Graph headroom context:

- TARGET 09.0 graph free memory moved from **55.21 GiB** before capture to
  **36.42 GiB** after capture: **18.78 GiB** free-memory delta.
- TARGET 08.07 per-bucket graph private-pool attribution showed the first bucket
  dominates: `[16]` cost about **18.828 GiB**, then buckets 8/4/2/1 added only
  about `0.057/0.057/0.049/0.047 GiB`.
- Persistent cache reductions improve free headroom before graph capture, but
  do not by themselves prove a smaller graph private pool.

## SGLang-Aligned SWA Lifecycle Memory Ledger

SGLang source evidence:

- `BaseSWAKVPool` exposes a separate `swa_kv_pool` and full-to-SWA mapping.
- `SWATokenToKVPoolAllocator` owns separate full and SWA allocators
  (`/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py:20`
  to `:78`) and can free SWA separately (`:341` to `:353`).
- `SWAComponent` stores translated SWA pool indices independently from full
  attention indices and tombstones SWA while full remains intact
  (`/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py:42`
  to `:68`).
- `free_swa_out_of_window_slots()` frees SWA outside the sliding window and
  keeps the eviction frontier page-aligned
  (`/workspace/sglang-main/python/sglang/srt/mem_cache/common.py:68` to
  `:112`).
- SGLang DSV4 pool construction separates `swa_kv_pool`, `c4_kv_pool`,
  `c128_kv_pool`, `c4_indexer_kv_pool`, and compression state ownership
  (`/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:438`
  to `:590`).

Mini cannot runtime-prove this lifecycle yet: TARGET 08.20 SWA tail retention
V1 is intentionally fail-closed because mini still derives several component
locations from full-token pages. The table below is therefore estimated.

Assumption: with `sliding_window=128` and `page_size=256`, one page-aligned SWA
tail page is the minimum retained SWA unit per active or retained branch.
Non-SWA C4/C128/indexer/state capacity remains at 128 pages.

| scenario | bound | current retained pages | tail SWA pages | tail SWA tokens | BF16 lifecycle GiB | remaining SWA-only FP8 value GiB | assumption |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_1024_bs4 | low | 124 | 4 | 1024 | 1.001 | 0.018 | bs4 long prompts; one page-aligned SWA tail per active/retained branch |
| historical_4096_1024_bs4 | high | 124 | 4 | 1024 | 1.001 | 0.018 | bs4 long prompts; one page-aligned SWA tail per active/retained branch |
| serving_mixed_112req_wave16 | low | 106 | 16 | 4096 | 1.127 | 0.072 | wave16 active-tail lower bound; short no-share branch retention upper bound |
| serving_mixed_112req_wave16 | high | 106 | 106 | 27136 | 2.071 | 0.477 | wave16 active-tail lower bound; short no-share branch retention upper bound |
| prefix_multi_112req_wave16 | low | 87 | 8 | 2048 | 1.043 | 0.036 | eight shared 512-token prefixes lower bound; wave16 active tails upper bound |
| prefix_multi_112req_wave16 | high | 87 | 16 | 4096 | 1.127 | 0.072 | eight shared 512-token prefixes lower bound; wave16 active tails upper bound |
| serving_mixed_256req_wave64_est | low | estimated | 64 | 16384 | 1.631 | 0.288 | synthetic higher-concurrency wave64 serving estimate |
| serving_mixed_256req_wave64_est | high | estimated | 64 | 16384 | 1.631 | 0.288 | synthetic higher-concurrency wave64 serving estimate |

Interpretation:

- In the long/prefix workloads, independent lifecycle alone recovers most of
  the current over-retained SWA memory. That makes SWA-only FP8 a much smaller
  follow-up.
- The serving mixed short-prompt upper bound stays high because many requests
  fit into one page. That is not strong evidence for FP8; it is the case where
  there is little out-of-window SWA to free.
- At higher concurrency, SWA-only FP8 can regain some capacity value
  (`0.288 GiB/rank` at 64 tail pages), but the first-order step is still proving
  independent lifecycle and measuring real tail-page occupancy.

## FP8 Layout Parity Notes

SGLang/vLLM aligned layout:

- SWA/C4/C128 MLA cache slot: **448 noPE FP8 bytes + 64 RoPE BF16 dims
  (128 bytes) + 8 UE8M0 scale/pad bytes**.
- Page token-data stride is **576 bytes**; scale bytes are stored after token
  data and page storage is padded to a multiple of 576 bytes.
- C4 indexer cache: **128 FP8 bytes + 4 FP32 scale bytes** per slot, or optional
  FP4 indexer (`index_head_dim // 2 + 4`) in SGLang.
- Compression state remains BF16/FP32-like; do not quantize state in this route.
- RoPE tail remains BF16.

Source pointers:

- SGLang constants:
  `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py:12`
  to `:21`.
- SGLang packed pool asserts `448 + 64*2 + 8 = 584 bytes/token` and pads pages:
  `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:94`
  to `:113`.
- SGLang C4 indexer uses `index_head_dim + 4`, or `index_head_dim // 2 + 4`
  for FP4:
  `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:248`
  to `:286`.
- SGLang fused SWA store path:
  `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:1016`
  to `:1058`.
- vLLM hardcodes the same FP8 sparse head byte formula:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:518`
  to `:524`.
- vLLM canonicalizes DeepSeek V4 KV cache to `fp8_ds_mla`:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1138`
  to `:1164`.
- vLLM selected-row cache utilities gather/dequantize FP8 K cache without a
  full-cache dequant path:
  `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py:7`
  to `:14`.

Answer to the explicit C4/C128 noPE question: **No, C4/C128 noPE should not
remain BF16 on the source-aligned FP8 route.** Keeping noPE BF16 is the current
mini-like precision path. It avoids FP8 quality risk, but then C4/C128 memory
benefit is small and the implementation is no longer aligned with the mature
SGLang/vLLM `fp8_ds_mla` layout.

## Speed Ledger

TARGET 09.4 measured a separated SWA packed store plus selected-row
gather/dequant harness on A100/sm80. It passed correctness and graph replay, but
was slower than the BF16 boundary.

| bs | selected rows | BF16 combined ms | FP8 separated combined ms | delta ms | graph workspace |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 0.044858 | 0.060692 | +0.015833 | 0.12 MiB |
| 2 | 256 | 0.042947 | 0.060492 | +0.017545 | 0.25 MiB |
| 4 | 512 | 0.044360 | 0.061223 | +0.016863 | 0.50 MiB |
| 8 | 1024 | 0.044981 | 0.061709 | +0.016728 | 1.00 MiB |
| 16 | 2048 | 0.045042 | 0.061205 | +0.016163 | 2.00 MiB |

Graph replay evidence from 09.4: captured buckets `1,2,4,8,16`, replay/eager
`5/0` for each bucket, max replay allocation delta `0 B`, capture allocation
delta `1024 B` per bucket, workspace preallocated.

| design | expected latency delta | evidence | risk note |
| --- | --- | --- | --- |
| BF16 baseline | 0 | runtime-proven promoted path | none |
| separated FP8 store + selected gather/dequant | +0.016 to +0.018 ms/boundary; worst about 0.75 ms if paid by all 43 layers | runtime-proven slice | too slow as production shape |
| SGLang-aligned fused store + selected-row gather/dequant | +0.006 to +0.012 ms/boundary estimate | estimated from removing store launch and keeping selected-row dequant | acceptable only as capacity opt-in until macro-proven |
| attention-integrated dequant | 0 to +0.006 ms/boundary estimate | source-derived plausible, not mini-proven | highest kernel and correctness risk |

HBM interpretation: FP8 reduces persistent cache bytes, but 09.4's separated
software encode/decode kernels are launch/arithmetic dominated at these bucket
sizes. A production path must either fuse norm/RoPE/packed-store, or integrate
dequant into sparse attention so the selected-row BF16 materialization does not
become another per-layer tax.

## ROI Decision Matrix

The table uses the requested 128-page current mini formula with the existing
additive indexer FP8 side cache as baseline (`2.302 GiB/rank`). "Equiv current
pages/tokens" is freed headroom divided by the current promoted
`18.42 MiB/page` page formula; it is not a claim that every row can simply add
that many logical pages without a new allocator policy.

For the SGLang lifecycle rows, the canonical value uses **16 tail SWA pages**,
matching the wave16 serving/prefix active-tail model. The lifecycle section
above shows the scenario range.

| row | persistent GiB/rank | graph headroom delta | equiv current pages/tokens | expected latency delta | quality/correctness risk | implementation scope |
| --- | --- | --- | --- | --- | --- | --- |
| current mini BF16 + additive indexer FP8 side | 2.302 | +0.000 GiB | 0.0 pages / 0 tokens | baseline | low; runtime-proven promoted path | none |
| current mini + SWA-only FP8 | 1.726 | +0.576 GiB | 32.0 pages / 8199 tokens | +0.016 ms/boundary if separated; needs fusion | medium quality/latency; correctness slice passed | replace SWA cache only; keep C4/C128/indexer/state |
| current mini + full source-aligned MLA/indexer FP8 | 1.614 | +0.688 GiB | 38.3 pages / 9794 tokens | unknown; likely worse until fused/integrated | high; C4/C128/indexer/prefix ownership not integrated | SWA+C4+C128 MLA FP8 and indexer replacement |
| SGLang lifecycle + BF16 | 1.127 | +1.176 GiB | 65.4 pages / 16734 tokens | near baseline or slight metadata cost | medium correctness; lifecycle not runtime-proven in mini | independent SWA pool, 16 tail pages, BF16 cache dtype |
| SGLang lifecycle + SWA-only FP8 | 1.055 | +1.248 GiB | 69.4 pages / 17759 tokens | estimated +0.006-0.012 ms/boundary if fused store + selected gather | medium-high; combines lifecycle and FP8 | lifecycle plus FP8 SWA tail pool (16 pages) |
| SGLang lifecycle + broader MLA/indexer FP8 | 0.942 | +1.360 GiB | 75.6 pages / 19354 tokens | unknown; attention-integrated dequant may be needed | highest; broad source layout plus ownership rewrite | lifecycle plus SWA/C4/C128/indexer FP8 replacement |

At the same current-memory budget, current mini + SWA-only FP8 would fit about
`170` source-formula pages (`+42` pages) and full source-aligned MLA/indexer FP8
would fit about `182` pages (`+54` pages). That is less decisive than the
SGLang lifecycle result because lifecycle also fixes the over-retained SWA
ownership model that created much of the apparent SWA-only FP8 upside.

## One Next-Action Recommendation

Chosen next action: **write/run a SWA independent lifecycle target first**.

Scope for that target:

- Add an opt-in independent SWA component/pool for mini DSV4.
- Keep C4, C128, indexer, and compression state ownership independent and do
  not quantize them.
- Introduce a full-to-SWA mapping and page-aligned tail allocation/free model.
- Tombstone/free out-of-window SWA without freeing C4/C128/indexer/state.
- Preserve Route B component loc ownership and direct graph metadata behavior.
- Add counters for SWA tail pages, tombstoned pages, freed pages, retained
  C4/C128/indexer/state pages, and component-safe prefix hit length.
- Runtime-prove the four scenarios used here:
  `historical_4096_1024_bs4`, `serving_mixed_112req_wave16`,
  `prefix_multi_112req_wave16`, plus a higher-concurrency serving scenario
  such as `serving_mixed_256req_wave64_est` or a real wave64 macro variant.

Suggested flag for that lifecycle target:

```bash
--enable-dsv4-swa-independent-lifecycle
```

or, if kept as an env-gated kernel/runtime experiment:

```bash
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
```

Pass gates:

- Text/logit smoke passes with prefix on/off and component loc ownership on.
- Prefix hit/remap cases prove component-safe fixed-point match length.
- No stale component-row mismatch under `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1`.
- Graph replay stays zero-eager for buckets `1,2,4,8,16`.
- Runtime SWA tail-page counters match the expected page-aligned model.
- Long-prefix/prefix-wave scenarios recover at least **0.5 GiB/rank** persistent
  or logical current-page-equivalent headroom versus current mini.

Stop gates:

- Any full/SWA page free can invalidate C4/C128/indexer/state locs.
- Any prefix hit can mix tombstoned SWA with unavailable component state.
- Any graph replay path dynamically allocates SWA/component workspace.
- Tail-page counters show little memory reduction in the long/prefix workloads.
- The implementation requires a broad attention/kernel rewrite before it can
  prove lifecycle correctness.

## TARGET 09.5 Scope Decision

TARGET 09.5 is **deferred**, not revised or run now.

No FP8 cache E2E env flag is recommended for this stage. In particular, do not
introduce or run `MINISGL_DSV4_SM80_SWA_FP8_MLA_CACHE=1` as the next target
until the independent SWA lifecycle target reports real SWA tail occupancy and
macro latency.

Future 09.5 can be reopened only if the lifecycle target shows either:

- high-concurrency or short-branch workloads still keep enough SWA pages that
  SWA-only FP8 has at least about **0.25 GiB/rank** real value, or
- broader source-aligned MLA/indexer FP8 can be scoped with selected-row
  gather/dequant and no prefix/graph ownership rewrite beyond the lifecycle
  changes already proven.

## Raw Scripts And Commands

New ledger script:

```bash
python performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/scripts/build_fp8_cache_roi_sglang_lifecycle.py
```

Generated summaries:

```text
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/summaries/ledger.json
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/summaries/current_mini_memory_ledger.md
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/summaries/sglang_lifecycle_ledger.md
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/summaries/speed_ledger.md
performance_milestones/target09_fp8_cache_roi_sglang_lifecycle/summaries/roi_matrix.md
```

Consumed runtime evidence:

```text
performance_milestones/target09_low_precision_preflight/summaries/memory_ledger.json
performance_milestones/target09_minimal_fp8_kv_cache_slice/summaries/swa_packed_mla_slice_harness.json
```

Relevant prior commands:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 timeout 3600 torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_perf_matrix.py --model-path /models/DeepSeek-V4-Flash --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 --page-size 256 --num-pages 128 --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 --repeats 1 --warmup-repeats 0 --seed 20260705 --output-dir performance_milestones/target09_low_precision_preflight/raw/promoted_macro_default_four_scenarios --keep-going
```

```bash
python performance_milestones/target09_minimal_fp8_kv_cache_slice/scripts/swa_packed_mla_slice_harness.py \
  --warmup 20 \
  --iters 200 \
  --graph-replays 5 \
  --bench-buckets 1,2,4,8,16 \
  --graph-buckets 1,2,4,8,16 \
  --output performance_milestones/target09_minimal_fp8_kv_cache_slice/summaries/swa_packed_mla_slice_harness.json
```

Evidence-level map:

| item | level |
| --- | --- |
| Current mini component sizes | source-derived from mini pool code |
| Promoted additive indexer FP8 side cache active | runtime-proven from TARGET 09.0 active toggles |
| 09.0 KV/cache bytes, prefix retained pages, graph delta | runtime-proven |
| 09.4 FP8 packed MLA correctness and graph replay | runtime-proven |
| 09.4 separated FP8 microbench overhead | runtime-proven |
| SGLang/vLLM FP8 layout | source-derived |
| SGLang SWA tombstone/free lifecycle | source-derived |
| Mini independent SWA lifecycle memory scenarios | estimated |
| Fused store / selected-row gather speed model | estimated from 09.4 and source layout |

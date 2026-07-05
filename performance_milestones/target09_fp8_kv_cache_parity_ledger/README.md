# TARGET 09.3 DSV4 SM80 FP8 KV/cache parity ledger

Date: 2026-07-05

Scope: source parity and capacity ledger only. This milestone does not implement a complete FP8 KV/cache E2E path.

## Conclusion summary

TARGET 09.4 is a **GO**, but only for a minimal source-aligned FP8 KV/cache slice. The recommended first slice is a packed MLA cache kernel boundary for SWA rows: store/quantize BF16 `(448 noPE + 64 RoPE)` into the SGLang/vLLM packed layout, then gather/dequantize selected rows back to BF16 and compare against mini's current BF16 cache path. This is smaller than full C4/C128 integration while still covering the real store/quant + gather/dequant boundary that indexer-only FP8 does not cover.

No stop gate is hit:

- SM80-compatible source path exists: vLLM uses SM80 software-FP8 Triton/custom-op gather/dequant and fused compress/insert paths; SGLang has CUDA JIT fused norm/rope/store for DeepSeek V4 packed cache.
- The memory win is material: source-aligned full MLA+indexer replacement saves **0.667 GiB/rank** at `page_size=256`, `num_pages=128`.
- Dequant does not require full-cache dequant in the mature sources: vLLM and SGLang both gather selected rows and dequantize to BF16.
- Prefix/cache/graph integration has risks, but none are fundamental blockers if 09.4 starts with SWA-only packed MLA cache and keeps prefix component ownership unchanged.

The strongest rule for 09.4: align SGLang/vLLM packed layout first. A different mini layout should only be considered after the source-aligned slice is measured and shown to be worse for correctness or performance.

## Evidence levels

- **Runtime-proven**: mini baseline behavior from `performance_milestones/target09_low_precision_preflight/README.md`: BF16 cache capacity, graph memory headroom, prefix-cache retained pages, Route B component ownership, direct graph metadata buffers.
- **Source-derived**: SGLang/vLLM FP8 KV/cache behavior in the audited source files. No mini FP8 MLA attention path has been implemented or run in this milestone.
- **Mini source-derived**: existing mini code paths and dormant/partial FP8 indexer code.

## Source files audited

Mini:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/`

## Model and ledger assumptions

The ledger uses `/models/DeepSeek-V4-Flash/config.json`:

- `num_hidden_layers = 43`
- `head_dim = 512`
- `index_head_dim = 128`
- `qk_rope_head_dim = 64`
- `compress_ratios` has 44 entries in the raw config, but mini maps `compress_ratios[:num_layers]`. For `num_layers = 43`, this gives 21 C4 layers, 20 C128 layers, and 2 normal layers.
- mini target page config: `page_size = 256`, `num_pages = 128`.
- mini current cache dtype: BF16 for SWA, compressed C4/C128, C4 indexer, and compression state pools.

The FP8 target layout below is the source-aligned DeepSeek V4 layout:

- MLA/SWA/C4/C128 slot: `448` FP8 noPE bytes + `64` BF16 RoPE dims = `576` token-data bytes, plus `8` scale bytes. Seven scale bytes are used for `448 / 64` noPE blocks and one byte is padding.
- SGLang page allocation pads MLA pages to a multiple of `576` bytes. At mini's derived component page sizes this is effectively:
  - SWA page size 256: `149,760 B/page`, or `585 B/token`.
  - C4 component page size 64: `37,440 B/component-page`, or `585 B/C4 slot`.
  - C128 component page size 2: `1,728 B/component-page`, or `864 B/C128 slot`; the small C128 component page makes alignment padding visible.
- C4 indexer FP8 slot: `128` FP8 values + `4` FP32 scale bytes = `132 B/C4 slot`.
- Compression state pools remain BF16. They are not part of the FP8 KV/cache candidate set.

## SGLang/vLLM/mini component parity table

| Component | mini current | SGLang source | vLLM source | Conclusion |
| --- | --- | --- | --- | --- |
| SWA MLA KV | BF16 flat tensor `(layers, pages, page_size, 512)`. Stored by BF16 norm/rope/store kernels or fallback. Gather path reads BF16 cache. Runtime-proven baseline. | `DeepSeekV4SingleKVPool` can allocate packed FP8 MLA cache. Store path quantizes 448 noPE dims per 64-block, stores RoPE tail BF16, and writes UE8M0 scales. Source-derived. | `fp8_ds_mla` cache spec uses `torch.uint8`, alignment 576. Insert op quantizes KV and stores RoPE BF16. SM80 reference gather/dequant exists. Source-derived. | Can/should be FP8 first. This is the recommended 09.4 minimal slice. |
| C4 compressed MLA KV | BF16 flat component cache with Route B component-page ownership and page table cache. Store after compressor norm/rope. Runtime-proven as BF16. | Same packed FP8 MLA layout, but component page size is `page_size/4`. Source-derived. | Fused compress/norm/rope/insert sparse-attn path stores compressed KV in `fp8_ds_mla`. Source-derived. | Can be FP8 after SWA slice passes. Route B component page mapping makes it higher risk than SWA. |
| C128 compressed MLA KV | BF16 flat component cache with `page_size/128 = 2` slots per component page. Runtime-proven as BF16. | Same packed FP8 MLA layout, but tiny component pages have proportionally larger 576-byte alignment padding. Source-derived. | Same compressed KV gather/dequant family. Source-derived. | Can be FP8, but memory win is small because the C128 slot density is already high and state dominates. Not first slice. |
| RoPE tail | Stored BF16 in mini's 512-wide BF16 KV rows. Runtime-proven. | Explicitly kept in original dtype/BF16 inside packed cache; not FP8-quantized. Source-derived. | Same `fp8_ds_mla` convention: noPE FP8, RoPE BF16. Source-derived. | Must remain BF16 for source parity. Do not quantize RoPE tail in mini FP8 cache. |
| C4 indexer cache | BF16 component cache `(c4_layers, slots, 128)`. Optional `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE` adds a uint8 paged side cache `(128 FP8 + 4 FP32 scale)`, but BF16 cache is still allocated. Mini source-derived. | `DeepSeekV4IndexerPool` uses `128` FP8 bytes + `4` FP32 scale bytes per C4 slot. Source-derived. | `DeepseekV4Indexer` uses the same 132-byte FP8 indexer slot family, with optional FP4 variants. Source-derived. | Can be FP8, but mini must replace the BF16 indexer cache rather than add a side cache if the goal is memory saving. Indexer-only is not enough for 09.4 because it skips attention gather/dequant. |
| C4/C128 compression state | BF16 ring/state pools: C4 ring size 8, C128 ring size 128, indexer state separate. Runtime-proven as BF16. | `CompressStatePool` stores KV/score state in `c4_state_dtype`/`c128_state_dtype`, not packed FP8 cache. Source-derived. | Compression kernels keep state/math separate from packed KV output; state is not the FP8 cache target. Source-derived. | Must remain BF16/FP32-like for correctness. Do not include in FP8 memory-saving claim. |
| Prefix-cache component ownership metadata | mini owns full pages plus Route B component pages for C4/C128/indexer/state. Prefix retained component bytes are already reported. Runtime-proven. | Source pool has separate SWA/C4/C128/indexer pools and state pools. Source-derived. | Uses cache specs and block tables per sparse scope. Source-derived. | Metadata stays integer/BF16-independent, but it must carry a cache layout/version tag once FP8 and BF16 coexist. |
| Direct graph metadata buffers | mini captures page tables, sparse indices, component write locs, and can regenerate direct index metadata for C4 groups. Runtime-proven. | Not a direct equivalent, but source kernels assume stable page/block tables and strides. Source-derived. | SM80 reference paths avoid full-cache dequant but require stable workspace and block strides. Source-derived. | Metadata dtype does not change, but FP8 gather/store kernels must use graph-stable strides and preallocated workspace. |
| Scale/workspace | mini current BF16 KV cache has no MLA scales. Optional indexer FP8 side cache stores FP32 scale bytes. Mini source-derived. | MLA cache uses UE8M0 scale byte per 64 noPE dims, padded to 8 bytes/token; indexer uses FP32 scale. Source-derived. | Same `fp8_ds_mla` per-token scale convention; generic KV scale checkpoints are not used for this dynamic per-token format. Source-derived. | MLA scales are part of persistent cache cost. Decode/prefill BF16 gather workspaces are temporary and must be graph-stable. |

## Component map

| Component | Current mini dtype/layout | Source-aligned FP8 dtype/layout | Scale format | Quantize/store boundary | Gather/dequant boundary | Decode/prefill behavior | SM80 status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWA KV | BF16, `(43, 128 pages, 256 tokens, 512 dims)` | uint8 packed pages: 448 noPE FP8 bytes + 64 RoPE BF16 dims + 8 scale bytes per token, padded page stride | UE8M0 byte scales for seven 64-wide noPE blocks, one pad byte | mini today: BF16 norm/rope/store. Source target: fused norm/rope/quant/store. | Source target: selected-row gather/dequant to BF16, not full-cache dequant. | Decode uses selected SWA rows; prefill may gather a BF16 workspace. | Source-compatible: vLLM SM80 reference gather/dequant; SGLang CUDA JIT store. | Runtime-proven BF16; FP8 source-derived. |
| C4 compressed KV | BF16, component page table, 64 C4 slots per full page | Same packed MLA slot, component page size 64 | UE8M0 + pad | After C4 compressor emits normalized/roped KV. | Gather selected compressed rows by component page table, dequant to BF16. | Decode selected top-k C4 rows; prefill uses compressed-scope gather workspace. | Source-compatible but needs Route B page-table correctness. | Runtime-proven BF16; FP8 source-derived. |
| C128 compressed KV | BF16, component page table, 2 C128 slots per full page | Same packed MLA slot, component page size 2, source-aligned page padding | UE8M0 + pad | After C128 compressor emits normalized/roped KV. | Gather selected C128 rows by component page table, dequant to BF16. | Decode selected top-k C128 rows; prefill uses compressed-scope gather workspace. | Source-compatible, small component page increases padding overhead. | Runtime-proven BF16; FP8 source-derived. |
| RoPE tail | BF16 inside 512-wide rows | BF16 bytes embedded after 448 FP8 noPE bytes | None | RoPE is applied before store and tail is copied/stored BF16. | Copied as BF16 during gather/dequant. | Same in decode and prefill. | Source-compatible. | Source-derived. |
| C4 indexer KV | BF16 `(21, slots, 128)`, optional additive uint8 side cache | uint8 `128` FP8 values + `4` FP32 scale bytes per C4 slot | FP32 scale bytes, not UE8M0 | Indexer norm/rope/hadamard output is quantized/stored per C4 slot. | Indexer logits dequant cache rows inside logits/select kernel. | Used for top-k selection before sparse attention. | mini has a CUDA Triton paged side-cache path; not memory-saving until BF16 buffer is removed. | Mini source-derived. |
| C4/C128/indexer state | BF16 state rings and state pages | Keep BF16/FP32-like | None for cache FP8 | Updated by compressor/indexer state logic, not packed KV store | Read by compressor/indexer logic, not sparse attention cache gather | Persistent state for decode/prefill compression | Must stay BF16/FP32-like. | Runtime-proven BF16; source-derived no-FP8 conclusion. |
| Prefix component metadata | int/page-handle ownership maps and component page tables | Same metadata, plus layout/version guard required | N/A | Allocator/prefix insert must reserve the matching component pages | Prefix replay must map to the same component pages and cache layout | Prefix hits retain pages across requests | Compatible if layout tags prevent BF16/FP8 mixing. | Runtime-proven BF16; FP8 risk source-derived. |
| Direct graph metadata | captured tensors for page tables, sparse indices, component write locs | Same metadata; FP8 kernels use different cache strides | N/A | Replay copies write locs and page tables before store | Gather/dequant uses captured page tables and fixed strides | Capture buckets must allocate stable BF16 gather workspace | Compatible if no dynamic allocation or eager fallback in capture. | Runtime-proven metadata; FP8 source-derived. |

## Capacity ledger

### Current mini BF16 cache

Per full page (`256` tokens):

| Component | Formula | Bytes/page | Bytes/token |
| --- | ---: | ---: | ---: |
| SWA KV | `43 * 256 * 512 * 2` | 11,272,192 | 44,032 |
| C4 compressed KV | `21 * 64 * 512 * 2` | 1,376,256 | 5,376 |
| C128 compressed KV | `20 * 2 * 512 * 2` | 40,960 | 160 |
| C4 indexer KV | `21 * 64 * 128 * 2` | 344,064 | 1,344 |
| C4/C128/indexer state | ring/state pools | 6,103,040 | 23,840 |
| **Total** |  | **19,136,512** | **74,752** |

Current mini capacity:

- `18.250 MiB/page`
- `2.281 GiB/rank` at `128` pages
- `32,768` tokens/rank at `page_size=256`, `num_pages=128`

This is the exact source-formula persistent cache estimate for mini's current `compress_ratios[:43]` mapping. It is consistent with the previous runtime-reported rounded/allocated `~2.32 GiB/rank` BF16 KV/cache pool.

### Source-aligned FP8 scenarios

| Scenario | What changes | Bytes/page | Bytes/token | GiB/rank at 128 pages | Saved GiB/rank | Extra pages at same memory | Extra tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWA-only FP8 | Only SWA MLA cache becomes packed FP8/BF16 RoPE | 14,304,000 | 55,875 | 1.705 | 0.576 | 43 | 11,008 |
| MLA-only FP8 | SWA + C4 + C128 MLA caches become packed FP8/BF16 RoPE; indexer remains BF16 | 13,707,584 | 53,545.25 | 1.634 | 0.647 | 50 | 12,800 |
| Full source-aligned replacement | MLA-only plus C4 indexer BF16 replaced by 132-byte FP8 indexer slots | 13,540,928 | 52,894.25 | 1.614 | 0.667 | 52 | 13,312 |

Notes:

- The table includes source-aligned MLA page padding. Without the C128 tiny-page padding, the full replacement would look slightly better, but that would diverge from the SGLang/vLLM layout before we have evidence to justify it.
- Existing mini `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE` is not counted as a saving by itself because it currently adds a side cache while keeping the BF16 indexer cache allocated. In that additive mode it increases persistent cache memory by `693 B/token` or `177,408 B/page`.
- Full source-aligned replacement saves `21,857.75 B/token`, or `5.336 MiB/page`.

### Scale and padding overhead

Persistent scale/padding costs are already included above:

| Cost | Bytes/page | GiB/rank at 128 pages | Notes |
| --- | ---: | ---: | --- |
| MLA UE8M0 scales + C4 indexer FP32 scales | 104,512 | 0.0125 | Seven MLA scale bytes plus one pad byte per MLA slot; four indexer scale bytes per C4 indexer slot. |
| Source-aligned MLA page padding | 23,552 | 0.0028 | Mostly SWA/C4 one byte per slot plus visible C128 small-page padding. |

Temporary workspace is not included in persistent cache capacity. The source-aligned path needs BF16 gather/dequant workspace:

- Decode workspace is proportional to selected sparse rows: roughly `batch * selected_rows * 512 * 2` bytes plus masks/metadata.
- Prefill workspace in vLLM is chunked and can be much larger because it gathers compressed and SWA scopes into BF16 buffers.
- For mini CUDA graph capture, any workspace must be allocated per graph bucket before capture and reused during replay. Dynamic allocation or eager fallback inside replay should be treated as a stop condition for promotion, not for this source ledger.

### CUDA graph headroom impact

Previous runtime baseline:

- Free memory before graph capture: `55.21 GiB`
- Free memory after graph capture: `36.42 GiB`
- Graph capture memory delta: `18.78 GiB`
- BF16 KV/cache capacity: `2.281 GiB/rank` by exact source formula; previous runtime report rounded/allocated this as `~2.32 GiB/rank`

If the same `128` pages are kept and full source-aligned FP8 replacement is used, expected persistent cache allocation drops to `1.614 GiB/rank`, freeing **0.667 GiB/rank** of graph headroom. The CUDA graph capture delta itself is not expected to shrink unless graph-captured attention workspaces are also reduced.

If the saving is reinvested into capacity instead, the same memory budget holds `180` FP8 pages instead of `128`, i.e. **52 additional pages** or **13,312 additional tokens**. In that mode graph headroom is roughly unchanged, but prefix/cache pressure improves by having more pages available.

### Prefix-cache retained pages

Using the full source-aligned replacement, each retained page saves `5.336 MiB`.

| Previous run | Retained pages | BF16 retained GiB | FP8 retained GiB | Saved MiB |
| --- | ---: | ---: | ---: | ---: |
| historical_4096_128_bs4 | 64 | 1.141 | 0.807 | 341.5 |
| historical_4096_1024_bs4 | 124 | 2.210 | 1.564 | 661.7 |
| serving_mixed_112req_wave16 | 106 | 1.889 | 1.337 | 565.7 |
| prefix_multi_112req_wave16 | 87 | 1.551 | 1.097 | 464.3 |

Interpretation: FP8 cache directly increases retained-prefix affordability, but retained page accounting must be layout-aware. Prefix metrics that currently multiply retained pages by BF16 component bytes would overstate FP8 retained memory after migration.

## Quant/dequant boundary map

| Boundary | mini today | SGLang/vLLM source target | 09.4 implication |
| --- | --- | --- | --- |
| SWA norm/rope/store | BF16 fused or fallback store writes `512` BF16 dims. | Quantize 448 noPE dims to FP8 per 64-block, store RoPE tail BF16, write UE8M0 scales. | First kernel slice should implement this packed store for SWA rows only. |
| SWA sparse gather | BF16 attention path reads/gathers BF16 rows. | Gather selected packed rows and dequant noPE to BF16; copy RoPE BF16; avoid full-cache dequant. | First slice must include selected-row gather/dequant and compare against BF16 gathered rows. |
| C4/C128 compress store | BF16 compressor output is stored into component caches. | Fused compress/norm/rope/quant/insert for compressed KV. | Defer until SWA slice passes; Route B component page tables must drive loc mapping. |
| C4/C128 sparse gather | BF16 C4/C128 caches are selected through component page tables. | Gather/dequant selected compressed rows from packed component pages. | Second slice should cover one C4 component path before C128 because C4 has meaningful memory and top-k traffic. |
| C4 indexer store | BF16 indexer cache plus optional additive FP8 side cache. | 128 FP8 values + FP32 scale per C4 slot. | Do not use as the only 09.4 slice; it lacks attention gather/dequant coverage. Later, convert from additive side cache to replacement for memory savings. |
| RoPE tail | BF16. | BF16. | Keep BF16; any FP8 RoPE tail proposal diverges from both sources. |
| Compression state update | BF16 state pools. | Keep state outside packed FP8 cache. | No FP8 state in 09.4. |

## Prefix/cache/graph correctness risk table

| Risk area | Failure mode | Severity | Evidence | Required guard before promotion |
| --- | --- | --- | --- | --- |
| Prefix cache layout mixing | A prefix entry retained under BF16 cache layout is replayed against FP8 packed buffers, or vice versa. | High | mini prefix runtime is proven for BF16; FP8 coexistence is source-derived only. | Add a cache-layout/version tag to retained component ownership metadata and invalidate/miss on mismatch. |
| Route B component ownership | FP8 C4/C128/indexer store uses full-token locs instead of component-page locs, corrupting component pages under prefix reuse. | High | Route B ownership is runtime-proven for BF16; FP8 component store is not integrated. | Derive FP8 store locs from `component_loc_ownership` and component page tables, not from `full_loc / ratio` shortcuts. |
| SWA tail retention | SWA window/tail indices point at rows that were not written, were padded, or were written with invalid locs during graph replay. | High | SWA tail retention is runtime-proven for BF16. Source FP8 kernels assume valid loc/page mapping. | SWA-only slice must test tail-heavy decode, skipped/padded rows, and prefix reuse with the same `out_loc` semantics as BF16. |
| Direct graph metadata buffers | Graph replay copies page tables/indices correctly but FP8 gather uses a different stride or allocates temporary workspace dynamically. | High | Direct graph metadata buffers are runtime-proven for BF16. vLLM SM80 paths show gather/dequant can be capture-sensitive. | Preallocate BF16 gather workspace per graph bucket; store packed cache stride constants in graph-stable metadata; no dynamic allocation in replay. |
| Scale format mismatch | MLA code reads indexer FP32 scale bytes as UE8M0 or reads MLA UE8M0 bytes as FP32 scale. | Medium | SGLang/vLLM use two different scale formats: MLA UE8M0, indexer FP32. | Keep separate typed helpers and layout constants for MLA packed cache vs indexer packed cache. |
| Full-cache dequant fallback | A correctness fallback dequantizes entire cache pages before attention, erasing memory/time gains. | Medium | vLLM/SGLang source paths have selected-row gather/dequant. | Promotion requires selected-row gather/dequant; full-cache dequant may only exist as a test/reference path. |
| Additive indexer FP8 cache | Enabling mini indexer FP8 increases memory because BF16 indexer buffer remains allocated. | Medium | Mini source-derived. | Count additive mode as a correctness experiment only. Memory ledger can count indexer FP8 only after BF16 indexer allocation is removed or gated off. |
| C128 padding surprise | C128 component page size is only 2, so source-aligned 576-byte page padding eats part of the theoretical win. | Low | Source-derived from SGLang page alignment. | Keep ledger source-aligned; if optimizing later, prove a divergent C128 layout is correct and faster. |

## Recommended minimal FP8 KV/cache slice for TARGET 09.4

Recommended 09.4 slice: **SWA packed MLA cache store + selected-row gather/dequant parity harness**, then optional mini integration behind an off-by-default flag.

Why this slice:

- It covers the real boundary that matters: BF16 KV row -> packed FP8 cache -> BF16 gathered row for attention.
- It uses the highest-value memory component. SWA-only FP8 saves **0.576 GiB/rank**, about 86% of the full source-aligned saving.
- It avoids Route B C4/C128 component ownership as the first integration risk.
- It does not depend on replacing mini's C4 indexer BF16 cache.
- It aligns directly with vLLM/SGLang MLA packed layout: 448 noPE FP8 + 64 RoPE BF16 + 8 scale bytes, page stride padded to 576-byte alignment.

Suggested 09.4 acceptance checks:

1. Kernel-level parity: random BF16 rows, page tables, and sparse indices; packed store followed by selected-row gather/dequant; compare against source/reference dequant with max/mean error and top-k stability.
2. mini SWA metadata parity: use real `out_loc`, `page_table`, `swa_page_indices`, and graph bucket shapes from the current DSV4 backend.
3. Capture safety: run the slice under CUDA graph replay with preallocated output workspace and no dynamic allocation.
4. Text/logit smoke: enable SWA-only FP8 for a narrow decode-only path and compare against BF16 route on the existing prefix/Route B smoke prompts.
5. Memory accounting: verify allocated cache bytes drop by the SWA-only ledger amount before attempting C4/C128/indexer replacement.

Do not choose the existing mini C4 indexer FP8 side cache as the main 09.4 slice. It is useful evidence that mini can store and read a paged FP8 side cache on SM80, but it does not exercise the attention MLA gather/dequant boundary and it currently increases memory unless the BF16 indexer cache is removed.

## 09.4 decision

Decision: **GO to TARGET 09.4**.

Allowed scope:

- Implement or port only the minimal SWA packed MLA store + selected-row gather/dequant slice first.
- Keep RoPE tail BF16.
- Keep C4/C128/indexer state BF16.
- Keep prefix/cache ownership unchanged for the first harness; do not migrate retained prefix pages in the same step.
- Treat C4/C128 compressed FP8 and indexer replacement as follow-up slices gated on SWA slice parity and graph-safety results.

Stop gates not hit in 09.3:

- No SM80-compatible path: **not hit**.
- Saved memory too small after scales/workspace: **not hit**; full persistent saving is `0.667 GiB/rank`, SWA-only is `0.576 GiB/rank`.
- Dequant requires full-cache dequant: **not hit**; source paths gather/dequant selected rows.
- Prefix/graph incompatibility: **not hit**, but remains the main 09.4 promotion risk.

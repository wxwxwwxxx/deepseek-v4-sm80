# TARGET 07.50: FP8 Cache/Indexer Precision Lane

## Status

Measured on 2026-07-01.  This target is an explicit opt-in precision/cache
experiment.  The default mini-sglang DeepSeek V4 path remains exact
`bf16_flat` for attention, compressed cache, and indexer cache.

Decision: stop the narrow FP8 indexer-first slice.  It is graph-safe and passes
text quality smoke, but it is slower than the exact bf16 control on microbench
and 4096/128 macro.  Because the `>=5%` macro gate was not met, 4096/1024 and a
fresh graph-node profile were not run.

Baseline exact stack from TARGET 07.40/07.395:

| Workload | Exact mini output tok/s | vLLM control output tok/s |
| --- | ---: | ---: |
| 4096/128/bs4 | `38.9379` | `82.2825` |
| 4096/1024/bs4 | `68.8097` | `202.0342` |

Prior evidence used here:

- TARGET 07.40: decode split-K gather/split/combine is no longer a top
  bottleneck; the largest exact buckets are runtime/copy/index, legacy prefill
  sparse, elementwise graph nodes, indexer/cache/topk, and projection GEMM.
- TARGET 07.41: replay metadata copy improved its microbench but did not move
  macro throughput, so more local exact metadata copy polish is not justified.
- TARGET 07.42: the strongest evidence-backed remaining mismatch is vLLM's
  `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, and FP8 indexer cache lane.
- TARGET 07.43: vLLM aux-stream overlap and persistent topk ablations were
  below the `5%` decision bar; CUDA graph is mandatory for vLLM, but mini
  already uses decode graph replay, so graph-node profiling is explanatory
  evidence only after an FP8 slice wins.

## Precision Boundary Map

vLLM source root: `/workspace/vllm-dsv4-docker`.

mini source root: `/workspace/mini-sglang`.

| Boundary | vLLM behavior | mini exact behavior | Planned 07.50 behavior | Deviation from vLLM? |
| --- | --- | --- | --- | --- |
| Engine/cache mode | DeepSeek V4 resolves to `deepseek_v4_fp8`; `DeepseekV4MLAAttention` treats `auto` as `fp8`, asserts FP8 KV cache, and canonicalizes non-`fp8_ds_mla` FP8 cache to `fp8_ds_mla` in `vllm/model_executor/layers/deepseek_v4_attention.py:1146-1164`. | `DeepSeekV4KVCache` default policy is `storage_dtype=torch.bfloat16`, `layout="bf16_flat"`, `indexer_layout="bf16_flat"` in `python/minisgl/kvcache/deepseek_v4_pool.py:24-36`; all SWA/C4/C128/indexer buffers are allocated as bf16 in `deepseek_v4_pool.py:190-206`. | Add a separate opt-in variant/env flag for the FP8 indexer/cache lane.  Do not reuse exact variant names and do not mutate default policy. | No for opt-in selection; yes intentionally for default, which remains exact bf16. |
| Activations kept bf16/fp32 | vLLM FlashMLA sparse backend supports bf16 compute inputs (`flashmla_sparse.py:93-100`).  MLA q concat workspace is bf16 when using `fp8_ds_mla` (`flashmla_sparse.py:759-767`).  Fused compressor math performs softmax/RMSNorm/RoPE in fp32, casts quant inputs through bf16 before FP8 quant (`fused_compress_quant_cache.py:471-487`, `529-533`). | mini q/KV norm/RoPE and cache stores keep bf16 tensors, with fp32 accumulation inside fallbacks and Triton kernels (`python/minisgl/kernel/deepseek_v4.py:1267-1338`, `python/minisgl/kernel/triton/deepseek_v4.py:884-948`). | Preserve bf16/fp32 compute around query projection, RMSNorm, RoPE, logits accumulation, topk metadata, and final sparse decode.  Quantize only explicit indexer-cache and later KV-cache storage boundaries. | No. |
| Indexer Q quantization | `fused_indexer_q_rope_quant` emits `(T,H,128)` `float8_e4m3fn` for FP8 Q.  On SM80 reference kernels it first emits fp32 scaled Q, then casts to FP8 because Triton lacks native `tl.float8e4nv` there (`fused_indexer_q.py:472-507`). | mini exact uses `DSV4Indexer.prepare_bf16_query` plus `indexer_q_rope_hadamard_bf16_fallback`; `DSV4AttentionBackend.select_indexer` passes bf16 q to `indexer_select_bf16_fallback` (`python/minisgl/models/deepseek_v4.py:633-694`, `python/minisgl/attention/deepseek_v4.py:287-321`). | First slice will add a vLLM-style FP8 Q path for indexer selection only: RoPE/Hadamard as today where required, absmax scale over 128, FP8 E4M3 values, logits weights carrying the folded q scale. | No in math intent.  The first implementation may use mini-owned Triton/software FP8 encode rather than vLLM custom op; record if outside-kernel quantization is used. |
| Q scale and weights | vLLM FP8 path does not pass a separate q scale to `SparseAttnIndexer`: comments and assertions state q scale is folded into `weights`; `q_scale must be None when use_fp4_cache=False` (`sparse_attn_indexer.py:158-163`).  The fused Q kernel documents `weights_out = weights * q_scale * softmax_scale * head_scale` (`fused_indexer_q.py:444-454`) and returns `weights_out` (`fused_indexer_q.py:565-586`). | mini exact passes model indexer weights separately; logits compute `relu(dot(q,k)) * weights` then sums heads (`deepseek_v4.py:1772-1800`). | Match vLLM FP8 semantics: no standalone FP8 Q scale tensor in the logits call; fold per-token/head q scale into the weights that feed FP8 logits. | No. |
| Indexer K/cache quantization and storage | vLLM FP8 indexer cache stores `128` FP8 bytes plus one `float32` scale per token/head: workspace shape is `(T, head_dim)` FP8 plus `(T,4)` uint8 scale bytes (`sparse_attn_indexer.py:50-68`), and the fused indexer cache insert documents cache block layout `[0, bs*128)` FP8 data plus `[bs*128, +bs*4)` float32 scales (`fused_compress_quant_cache.py:686-693`).  It quantizes all 128 dims as one block, scale `2^ceil(log2(absmax/448))`, stores scale as float32 bytes (`fused_compress_quant_cache.py:798-824`). | mini exact `store_indexer` calls `compress_norm_rope_store_fallback(..., cache_type="indexer")` and stores bf16 rows into `kvcache.indexer_cache(layer_id)` (`attention/deepseek_v4.py:232-285`, `kernel/deepseek_v4.py:2892-3056`).  `indexer_cache()` returns a flat bf16 `[c4_slots, index_head_dim]` buffer (`kvcache/deepseek_v4_pool.py:202-206`, `296-299`). | First slice will add an opt-in FP8 indexer cache side buffer with vLLM storage payload: data bytes `[slots,128]` plus scale bytes `[slots,4]` or an equivalent paged view that preserves the same per-slot layout.  Store path will quantize after existing indexer compression/norm/RoPE/Hadamard boundary. | Possible narrow deviation: mini may store data/scale as two tensors initially instead of vLLM's single paged byte tensor if that keeps the first slice small.  If used, logits/gather kernels must consume it without changing math and microbench the cost. |
| Indexer logits | vLLM prefill uses gathered FP8 K plus `fp8_mqa_logits_triton` on SM80 when DeepGEMM is unavailable (`sparse_attn_indexer.py:226-247`).  Decode packs/reshapes Q, views paged K cache as quantized cache, and on SM80 calls `fp8_paged_mqa_logits_triton` (`sparse_attn_indexer.py:280-350`). | mini exact calls `indexer_bf16_logits_fallback`, using Triton when `MINISGL_DSV4_SM80_INDEXER_BF16=1`, otherwise torch.  The Triton kernel loads bf16/float cache, computes fp32 dot per head, applies ReLU and weights, and writes fp32 logits (`kernel/deepseek_v4.py:1703-1837`, `kernel/triton/deepseek_v4.py:951-995`). | Add opt-in `fp8_paged_mqa`-style indexer logits over the FP8 indexer cache.  It should dequantize K on load using the per-slot scale, compute fp32 logits, apply ReLU and folded weights, and write fp32 logits for the existing topk transform. | No for math.  Kernel ownership differs: mini-owned Triton kernel instead of vLLM op. |
| Top-k indices | vLLM writes into a reusable `topk_indices_buffer`; decode may use `persistent_topk` when enabled (`sparse_attn_indexer.py:184`, `352-370`).  TARGET 07.43 showed persistent-topk ablation did not hurt macro. | mini exact uses `topk_transform_512_full_fallback` and propagates raw/page/full indices plus `topk_lens` into DSV4 metadata (`kernel/deepseek_v4.py:2509-2563`, `attention/deepseek_v4.py:322-339`). | Reuse existing mini topk/lens transform for first FP8 indexer slice.  Do not port persistent topk unless FP8 logits moves the bottleneck there. | Yes, deliberate and evidenced by TARGET 07.43. |
| MLA/SWA KV cache layout | vLLM DeepSeek V4 `fp8_ds_mla` token is 584 bytes: 448 FP8 NoPE bytes, 128 bf16 RoPE bytes, and 8 scale bytes with 7 `ue8m0` scales plus pad (`flashmla_sparse.py:81-89`).  `DeepseekV4FlashMLASparseBackend.get_kv_cache_shape` returns `(num_blocks, block_size, 584)` for `fp8_ds_mla` (`flashmla_sparse.py:150-167`).  SWA uses the same 584B/token shape (`sparse_swa.py:75-86`, `122-128`). | mini exact SWA/C4/C128 caches are flat bf16 tensors with semantic head dim 512 (`kvcache/deepseek_v4_pool.py:190-201`, `296-310`). | Defer full `fp8_ds_mla` cache until the indexer-first slice is measured or blocked.  If needed, add opt-in packed `[num_blocks, block_size, 584]` storage with 576B token stride and separate block scale tail matching vLLM's layout. | Not in first slice; planned later only if justified. |
| MLA/SWA gather/dequant into sparse decode | vLLM sparse decode is `compute_global_topk_indices_and_lens -> gather_dequant_two_scopes_with_mask -> _dsv4_sm80_sparse_attn_decode_triton` (`deepseek_v4_attention.py` imports these ops and TARGET 07.42/dispatch report verified this path).  `compute_global_topk_indices_and_lens` maps local topk to global slots and counts valid entries in one kernel (`cache_utils.py:974-1043`). | mini exact already adapted this boundary for bf16: global topk/lens plus bf16 gather/mask and split-K sparse decode; the sparse-only microbench is near vLLM's comparable probe (`0.2284 ms` vs `0.2258 ms`). | Keep bf16 split-K sparse decode for first indexer slice.  If implementing `fp8_ds_mla`, add gather/dequant before the existing bf16 split-K sparse decode instead of replacing sparse decode first. | No for algorithmic boundary; storage precision changes only under opt-in. |
| Dequant placement | vLLM moves dequant into fused/custom ops: indexer logits dequantizes FP8 K inside logits kernels; MLA packed cache dequant happens in gather/dequant before sm80 sparse decode. | mini exact has no cache dequant; kernels load bf16 directly. | First slice dequantizes indexer K inside the FP8 logits kernel.  Later `fp8_ds_mla` slice should dequantize inside gather/dequant before existing sparse decode. | No. |

## Minimal First Slice

The first implementation slice is the narrow FP8 indexer cache/logits lane:

1. Add a separate opt-in flag and named macro/text-smoke variant.
2. Add FP8 indexer K/cache storage for C4 indexer rows only, preserving the
   existing bf16 indexer cache and default path.
3. Add vLLM-style FP8 indexer Q quantization with q scale folded into weights.
4. Add FP8 paged indexer logits that consume the FP8 cache and write fp32
   logits into the existing topk/lens path.
5. Add synthetic correctness and quality probes: quant/dequant error,
   logit max/mean error, topk exact/overlap rates, and TP8 text smoke.
6. Run microbench plus 4096/128/bs4 macro.  Only if 4096/128 improves by at
   least `5%`, run 4096/1024/bs4 and a fresh mini graph-node profile.

If this slice is blocked by graph-capture layout or model integration, record
the blocker here and pivot to the smallest viable `fp8_ds_mla` cache
gather/dequant slice.

## Validation Plan

Required correctness:

- synthetic FP8 indexer quantize/dequantize/store tests against a bf16 source;
- FP8 indexer logits vs bf16 logits on controlled deterministic cases;
- top-k exact match and overlap rate vs exact bf16;
- default-path guard tests proving env unset preserves bf16 behavior;
- TP8 text smoke with English and Chinese prompts.

Required performance:

- indexer store/logits/topk microbench for bf16 vs FP8 lane;
- 4096/128/bs4 macro against the best exact mini stack;
- 4096/1024/bs4 macro only after a `>=5%` 4096/128 win;
- graph node profile only after a `>=5%` macro win, as explanatory evidence.

## Implementation

Implemented the narrow opt-in indexer slice under
`MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1`:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
  - adds an opt-in side cache for C4 indexer rows:
    `[c4_layers, c4_slots, index_head_dim]` uint8 FP8 payload and
    `[c4_layers, c4_slots, 4]` uint8 scale bytes;
  - default bf16 indexer cache allocation and access are unchanged.
- `python/minisgl/kernel/deepseek_v4.py`
  - adds FP8 indexer cache quant/dequant refs;
  - adds vLLM-style indexer Q FP8 quantization with q scale folded into
    weights;
  - adds FP8 paged indexer logits/select fallbacks and graph-safe Triton dispatch;
  - routes indexer store to the FP8 side cache only when the new flag is set.
- `python/minisgl/kernel/triton/deepseek_v4.py`
  - adds software E4M3 encode, FP8 indexer cache quant-store, and FP8 paged
    logits kernels.
- `python/minisgl/models/deepseek_v4.py` and
  `python/minisgl/attention/deepseek_v4.py`
  - connect the opt-in FP8 query/cache/logits lane;
  - keep the existing exact bf16/Hadamard lane when the flag is unset.
- `benchmark/offline/deepseek_v4_perf_matrix.py` and
  `benchmark/offline/deepseek_v4_text_smoke.py`
  - add a named FP8 indexer-cache variant:
    `v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`.

## Validation Results

Raw and summary artifacts:

- summary: `summaries/target0750_indexer_fp8_summary.json`
- microbench: `raw/mini_indexer_fp8_microbench.json`
- quick microbench: `raw/mini_indexer_fp8_microbench_quick.json`
- TP8 text smoke: `raw/tp8_text_smoke_idxfp8cache.json`
- FP8 4096/128 macro:
  `raw/dsv4_target0750_idxfp8cache_4096x128_bs4_np128/summary.json`
- exact control 4096/128 macro:
  `raw/dsv4_target0750_exactcontrol_4096x128_bs4_np128/summary.json`

Tests run:

```bash
python -m py_compile python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m black --check python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  performance_milestones/target07_fp8_cache_indexer_precision/scripts/mini_indexer_fp8_microbench.py

python -m pytest -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_indexer_fp8_quantized_logits_and_topk_match_reference \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

Unit/registry outcome: `36 passed`.  Note: this environment lacks
`pytest-cov`, so pytest was run with `-o addopts=''` to bypass the repository
coverage addopts.

Microbench, A100 sm80, `iters=10`, `warmup=3`:

| Case | BF16 logits ms | FP8 logits ms | BF16 select ms | FP8 select ms | Top-k overlap mean | Logit mean abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch1, history1024 | `0.1271` | `0.1573` | `0.3093` | `0.3182` | `0.9863` | `0.02284` |
| batch4, history2048 | `0.1246` | `0.2655` | `0.3037` | `0.3163` | `0.9829` | `0.02178` |
| batch16, history4096 | `0.3076` | `1.3072` | `0.3586` | `1.7368` | `0.9724` | `0.02142` |

FP8 indexer cache storage is smaller in the synthetic probe, but the first
mini-owned FP8 paged logits kernel is slower than the exact bf16 Triton logits
kernel, especially at batch16/history4096.  Cache dequant mean absolute error
was approximately `0.01796-0.01798`; max absolute error was `0.25`.

TP8 text smoke:

- status: `pass`
- prompts: 3/3 sane
- outputs:
  - `2 + 2 等于 4。`
  - `The sky is blue on a clear day.`
  - `杭州是风景如画的历史文化名城。`
- graph replay count: `9`
- eager decode count: `0`

4096/128/bs4 macro, TP8, page size 256, `num_pages=128`, repeats 3, warmup 1:

| Variant | Output tok/s | Decode tok/s | Mean TTFT s | Prefill tok/s | Eager decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact control, no FP8 indexer flag | `37.9237` | `79.8574` | `4.3114` | `2645.17` | `0` |
| FP8 indexer cache/logits opt-in | `29.6691` | `81.5617` | `6.7446` | `1620.46` | `0` |

The FP8 lane is `-21.77%` vs the same-run exact control and `-23.80%` vs the
historical best exact `38.9379` line.  It therefore fails the target's `>=5%`
continuation gate.

## Decision

Do not run 4096/1024 or a fresh mini graph-node profile for this slice.  The
macro loss is already explained by the microbench direction: the current
mini-owned FP8 paged indexer logits implementation is slower than the existing
bf16 Triton path, and the FP8 query/cache quantization adds prefill cost while
only slightly improving decode tok/s.

If continuing TARGET 07.50, the next reasonable fork is one of:

- replace the FP8 indexer logits kernel with a closer vLLM-style optimized
  paged logits kernel before retrying macro; or
- pivot to a minimal `fp8_ds_mla` cache/gather/dequant slice, since the
  isolated FP8 indexer side cache does not produce a standalone macro win.

# TARGET 05.5: DSV4 sm80 High-Performance Kernel R&D Plan

## Goal

Use the wrapper boundary created in TARGET 05 to replace DeepSeek V4 fused
kernel fallbacks with sm80/A100-friendly high-performance kernels. This target
is a research and engineering plan: every kernel must keep a safe fallback,
must be gated behind an explicit opt-in flag at first, and must record
correctness/performance evidence before any default promotion.

The main design choice for act quant sensitive kernels is to test two lanes:

- `bf16-direct`: remove activation quantization from the fast path, pass bf16
  tensors, and use sm80 Tensor Core paths where possible.
- `quantized-act`: keep fp8/fp4 activation or KV quantization, but dequant or
  apply scales inside the sm80 kernel before the tensor-core computation.

Default rule: implement and validate `bf16-direct` first. Add `quantized-act`
only as an opt-in comparison lane unless it clearly improves end-to-end
DeepSeek V4 performance or materially reduces memory without oracle drift.

## Primary References

- Current wrapper boundary: `python/minisgl/kernel/deepseek_v4.py`
- Current DSV4 model call sites: `python/minisgl/models/deepseek_v4.py`
- Current DSV4 attention backend: `python/minisgl/attention/deepseek_v4.py`
- Upstream JIT DSV4 kernels: `/workspace/sglang-main/python/sglang/jit_kernel/dsv4`
- Upstream DSV4 model: `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
- Upstream DSV4 attention backend: `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- Upstream DSV4 attention helpers: `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4`
- Old branch lessons: `git show dsv4:prompts/PERF_RESULTS.md`

## Kernel Matrix

| Wrapper / Kernel | Current Interface | Upstream Reference | sm80 Implementation Plan | Toggle / Variant |
| --- | --- | --- | --- | --- |
| `q_norm_rope_fallback` | `q[tokens, heads, dim]`, `positions[tokens]` -> in-place q | `fused_q_norm_rope`, `main_norm_rope.cuh` | Triton first: one program per token/head, RMSNorm + RoPE tail. Later TileLang/CUDA only if Triton is slow. | `MINISGL_DSV4_SM80_Q_NORM_ROPE=1` |
| `apply_rotary_tail` | `x[..., rope_dim]`, `positions` -> in-place x | `fused_rope_inplace` | Low-risk Triton RoPE shared by q/k/o inverse. Keep torch fallback for CPU/tests. | `MINISGL_DSV4_SM80_ROPE=1` |
| `k_norm_rope_cache_fallback` | `kv[tokens, dim]`, positions, cache locations -> rotated kv/cache write | `fused_k_norm_rope_flashmla` | Phase 1 `bf16-direct`: RMSNorm + RoPE + bf16 store. Phase 2 `quantized-act`: split nope fp8 + rope bf16 pack. | `MINISGL_DSV4_SM80_KV_BF16=1`, `MINISGL_DSV4_SM80_KV_FP8=1` |
| `store_swa_fallback` | `kv[tokens, dim]`, `out_loc[tokens]` -> cache write | `fused_store_cache` | Triton gather/scatter store for contiguous and paged locations. No quantization in v1. | `MINISGL_DSV4_SM80_STORE_CACHE=1` |
| `compress_forward_fallback` | `x[tokens, hidden]`, compressor weights/state -> compressed kv | `compress_forward`, `Compressor*Plan` | Keep Python planner first; replace pooling loop with Triton kernels for ratio 4 and 128. Split prefill/decode only if shapes require it. | `MINISGL_DSV4_SM80_COMPRESS=1` |
| `compress_norm_rope_store_fallback` | compressed kv + loc -> compressed cache | `compress_norm_rope_store`, `fused_norm_rope_inplace` | Fuse norm + optional RoPE + cache store. Start bf16-direct, then add fp8 cache pack if bandwidth matters. | `MINISGL_DSV4_SM80_COMPRESS_STORE=1` |
| `topk_transform_512_fallback` / v2 | index/logit rows -> 512-wide sparse indices | `topk_transform_512`, `plan_topk_v2` | Triton top-k/materialization tuned for sm80 shared memory. Avoid sm90/sm100 shared-memory assumptions. | `MINISGL_DSV4_SM80_TOPK=1` |
| `paged_mqa_attention_fallback` | q, cache, per-row indices -> attention output | DeepGEMM paged-MQA and TileLang indexer paths | Phase 1 bf16 sparse values kernel, decode first. Phase 2 fp8 KV dequant in-kernel. | `MINISGL_DSV4_SM80_PAGED_MQA_BF16=1`, `MINISGL_DSV4_SM80_PAGED_MQA_FP8=1` |
| `flash_mla_with_kvcache` | q + paged KV + metadata -> output | `sgl_kernel.flash_mla` | Treat FlashMLA as reference only unless sm80 binary support exists. Implement local sparse MLA path first. | experimental only |
| `fused_q_indexer_rope_hadamard_quant` | indexer q + weight + positions -> fp8 q + weights | `FusedQIndexerRopeHadamardQuantKernel` | Two lanes: bf16-direct indexer logits first; fp8 q quant only if bf16 bandwidth is bottleneck. | `MINISGL_DSV4_SM80_INDEXER_BF16=1`, `MINISGL_DSV4_SM80_INDEXER_FP8=1` |
| `fused_q_indexer_rope_hadamard_fp4_quant` | indexer q -> packed fp4 q + scale + weights | sm100-oriented fp4 path, DeepGEMM fp8/fp4 paged logits | Keep unsupported by default. Implement only after fp8 indexer works; require oracle gate because fp4 approximation risk is higher. | `MINISGL_DSV4_SM80_INDEXER_FP4=1` |
| `quantized_linear_ref` | x + fp8/fp4 weight + scale -> linear | DeepGEMM, FlashInfer, Marlin-like GEMM paths | First target MoE large shapes. Avoid naive small-shape TileLang FP4 as default. Prefer grouped route-aware fp4 dequant + MMA. | `MINISGL_DSV4_SM80_FP4_GEMM=1`, `MINISGL_DSV4_SM80_FP8_GEMM=1` |
| `wo_a_grouped_projection_fallback` | `o[tokens, groups, d]`, fp8 weight -> LoRA output | DeepGEMM `fp8_einsum` | First bf16 dequant-on-load Triton einsum. Then optional fp8 activation quant + grouped MMA. | `MINISGL_DSV4_SM80_WO_A_BF16=1`, `MINISGL_DSV4_SM80_WO_A_FP8=1` |
| `silu_and_mul_clamp_fallback` | gate/up activations -> SwiGLU output | `silu_and_mul_clamp` | Triton elementwise fusion. This is a low-risk early win. | `MINISGL_DSV4_SM80_SWIGLU=1` |
| `silu_and_mul_*_post_quant` | expert activations -> quantized activation | SGLang MoE post-quant kernels | Implement only with grouped fp4/fp8 GEMM work; post-quant alone likely adds overhead. | tied to GEMM toggles |
| `moe_gate_fallback`, `hash_topk_fallback`, `mega_moe_pre_dispatch_fallback` | router scores/ids -> weights, indices, grouped dispatch | `hash_topk`, `mask_topk_ids`, `mega_moe_pre_dispatch` | Replace Python expert route loops with exact route-aware grouping. Do not reuse graph capture dense invalid-route fallback as a default path. | `MINISGL_DSV4_SM80_MOE_ROUTE=1` |
| `hc_pre/post/head_fallback` | HC residual mix tensors -> mixed states | MHC helper paths | Lower priority. Add Triton HC split/Sinkhorn only after attention and MoE wins land. | `MINISGL_DSV4_SM80_HC=1` |
| `linear_bf16_fp32_fallback` | bf16/fp32 linear helper | `linear_bf16_fp32` | Optimize only if profiling shows HC/router matmul is visible. Otherwise leave torch. | no initial toggle |

## Implementation Order

1. Add wrapper-level runtime policy, no heavy kernels yet.
   - Add environment parsing helpers in `python/minisgl/kernel/deepseek_v4.py`.
   - Extend the inventory with candidate modes: `fallback`, `bf16_direct`, `fp8_act`, `fp4_act`.
   - Keep every new mode disabled by default.

2. P0 low-risk kernels.
   - `silu_and_mul_clamp_fallback`
   - `apply_rotary_tail`
   - `q_norm_rope_fallback`
   - `store_swa_fallback`
   - bf16 `paged_mqa_attention_fallback`

3. P1 attention/indexer structure.
   - `compress_forward_fallback`
   - `compress_norm_rope_store_fallback`
   - `topk_transform_512_fallback`
   - bf16 indexer logits path

4. P2 quantized and grouped compute paths.
   - fp8 KV pack/dequant
   - `wo_a` grouped projection
   - fp8 indexer quant
   - fp8/fp4 grouped GEMM candidates

5. P3 higher-risk approximate paths.
   - fp4 indexer
   - MoE post-quant fused activation
   - route-aware grouped MoE dispatch

6. P4 deferred helpers.
   - HC split/Sinkhorn helpers
   - metadata kernels that only matter after profiling confirms overhead

## Acceptance Ladder

Every kernel replacement must pass this ladder before it is considered landed:

1. Unit shape/dtype tests against the torch fallback.
2. Numeric tolerance tests on fixed seeds.
3. Existing DSV4 forward fallback tests.
4. DSV4 metadata/cache tests.
5. Microbench for isolated kernel speed and memory behavior.
6. End-to-end performance smoke with toggle on/off.
7. Optional oracle/top-k gate for fp8/fp4 approximation paths.

Default promotion requires:

- no correctness regression;
- no unsupported dependency on sm90/sm100-only binaries;
- at least one target DSV4 workload improves;
- no material regression on `decode_heavy_bs1`;
- the fallback path remains available and tested.

## Benchmark Commands

Correctness smoke:

```bash
pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/models/test_deepseek_v4_forward_fallback.py
```

Performance matrix template:

```bash
python -u benchmark/offline/deepseek_v4_perf_matrix.py \
  --output-dir /tmp/dsv4_sm80_target05_5 \
  --variants default bf16_direct quantized_act \
  --workloads interactive_bs1 decode_heavy_bs1 chat_batch4 long_context_bs1 prefix_cache_reuse \
  --keep-going
```

If the current branch does not yet contain `deepseek_v4_perf_matrix.py`, first
restore or recreate it from the old `dsv4` branch references listed in
TARGET 05, then keep this command shape as the standard reporting interface.

## R&D Completion Matrix

Codex should update this table whenever a replacement is attempted.

| Kernel | Status | Mode | Toggle | Correctness | Microbench | E2E Perf | Decision | Notes / Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `silu_and_mul_clamp_fallback` | implemented opt-in | bf16-direct Triton | `MINISGL_DSV4_SM80_SWIGLU` | passed: CUDA wrapper test plus default/all-toggle target smoke, 19 passed | 0.356 -> 0.044 ms, 8.04x | forward smoke only; perf matrix harness absent in current branch | keep opt-in; promotion candidate after E2E harness | `python/minisgl/kernel/triton/deepseek_v4.py`; `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `apply_rotary_tail` | implemented opt-in | bf16-direct Triton | `MINISGL_DSV4_SM80_ROPE` | passed: inverse/scaled RoPE CUDA parity plus target smoke | 0.416 -> 0.113 ms, 3.69x | forward smoke only; perf matrix harness absent | keep opt-in; good P0 candidate | Shared by q/k/o inverse RoPE; artifact `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `q_norm_rope_fallback` | implemented opt-in | bf16-direct Triton | `MINISGL_DSV4_SM80_Q_NORM_ROPE` | passed: RMSNorm+scaled RoPE CUDA parity plus target smoke | 0.485 -> 0.068 ms, 7.18x | forward smoke only; perf matrix harness absent | keep opt-in; good P0 candidate | One Triton program per token/head; artifact `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `store_swa_fallback` | implemented opt-in, negative perf | bf16-direct Triton scatter | `MINISGL_DSV4_SM80_STORE_CACHE` | passed: CUDA cache scatter parity plus target smoke | 0.033 -> 0.038 ms, 0.88x | forward smoke only; perf matrix harness absent | keep disabled by default; not a promotion candidate yet | Torch indexed store is already competitive for tested shape; artifact `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `paged_mqa_attention_fallback` | researched, not implemented | bf16-direct | `MINISGL_DSV4_SM80_PAGED_MQA_BF16` | fallback covered by target smoke | no replacement benchmark | not run | keep fallback; redesign metadata/layout first | FlashInfer exposes MLA wrappers, but current mini cache is bf16 flat/ragged Python list metadata; no safe drop-in paged MQA replacement yet |
| `flash_mla_with_kvcache` / `flash_mla_sparse_prefill` | researched, blocked | experimental | experimental only | fallback/unsupported paths covered | not run | not run | keep unsupported/reference-only | `sgl_kernel.flash_mla` imports, but mini lacks FlashMLA packed cache layout and metadata contract |
| `k_norm_rope_cache_fallback` | partially covered by RoPE opt-in | bf16-direct / fp8-act | `MINISGL_DSV4_SM80_KV_BF16`, `MINISGL_DSV4_SM80_KV_FP8` | RoPE subpath passed; dedicated KV norm/cache path not implemented | uses RoPE artifact only | forward smoke only | keep fallback for KV-specific path | Current wrapper only rotates `kv`; fp8 pack / FlashMLA cache write remains blocked on cache layout and no DSV4 sgl ops on sm80 |
| `compress_forward_fallback` | implemented opt-in | bf16-direct vectorized torch | `MINISGL_DSV4_SM80_COMPRESS` | passed: ratio-4 overlap CUDA parity plus target smoke | 163.730 -> 0.207 ms, 790.76x | forward smoke only; perf matrix harness absent | keep opt-in; highest P1 candidate | Replaced Python pooling loop with batched ratio grouping; artifact `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `compress_norm_rope_store_fallback` | partial store-only opt-in | bf16-direct | `MINISGL_DSV4_SM80_COMPRESS_STORE` | target smoke passed through fallback; store kernel shares cache-scatter test coverage | store subkernel only; no separate benchmark | forward smoke only | keep disabled; full norm+rope+store still pending | Current mini wrapper has no norm/rope args here, so only compressed/indexer store can use Triton scatter |
| `topk_transform_512_fallback` | implemented opt-in, neutral perf | indices padding Triton | `MINISGL_DSV4_SM80_TOPK` | passed: CUDA 512 materialization parity plus target smoke | 0.033 -> 0.033 ms, 1.00x | forward smoke only; perf matrix harness absent | keep disabled by default | This covers mini's pad/materialize fallback only, not upstream score/page-table top-k transform |
| `fused_q_indexer_rope_hadamard_quant` | researched, blocked | bf16-direct / fp8-act | `MINISGL_DSV4_SM80_INDEXER_BF16`, `MINISGL_DSV4_SM80_INDEXER_FP8` | existing bf16 indexer fallback covered by target smoke | no replacement benchmark | not run | keep bf16 fallback; implement only after paged indexer metadata is ready | `torch.ops.sgl_kernel` has no DSV4 ops; upstream JIT path needs SGLang C++ JIT headers and fp8 quant path is not useful enough on sm80 without downstream fp8 logits |
| `fused_q_indexer_rope_hadamard_fp4_quant` | blocked | fp4-act | `MINISGL_DSV4_SM80_INDEXER_FP4` | unsupported path test passes clear error | not run | not run | keep disabled | fp4 path is sm100/DeepGEMM-oriented; DeepGEMM fails to load locally with missing `libcudart.so.13`; requires oracle/top-k gate before any retry |
| `quantized_linear_ref` | researched, blocked for default | fp4/fp8 GEMM | `MINISGL_DSV4_SM80_FP4_GEMM`, `MINISGL_DSV4_SM80_FP8_GEMM` | fallback covered by target smoke | no new replacement benchmark | not run | keep dequant+torch fallback | A100 has no native fp8/fp4 tensor-core path; DeepGEMM unusable on this env; Marlin absent; grouped route-aware GEMM should be a separate larger effort |
| `wo_a_grouped_projection_fallback` | researched, not implemented | bf16-direct / fp8-act | `MINISGL_DSV4_SM80_WO_A_BF16`, `MINISGL_DSV4_SM80_WO_A_FP8` | fallback covered by target smoke | no replacement benchmark | not run | keep fallback; revisit after GEMM route work | Upstream uses DeepGEMM `fp8_einsum`; local DeepGEMM unusable, so a Triton dequant-on-load einsum remains pending |
| `silu_and_mul_*_post_quant` | blocked | fp8/fp4 post-quant | tied to GEMM toggles | unsupported paths raise clear errors | not run | not run | keep disabled | Post-quant alone would add overhead without a fused grouped GEMM consumer |
| `moe_gate/hash_topk/mega_moe_pre_dispatch` | researched, not implemented | routing | `MINISGL_DSV4_SM80_MOE_ROUTE` | fallback covered by target smoke | no replacement benchmark | not run | keep fallback; require exact route-aware grouping kernel | Upstream route kernels depend on SGLang JIT CUDA; capture-style dense invalid-route fallback should not be reused as default |
| `hc_pre/post/head_fallback` | deferred after research | bf16-direct | `MINISGL_DSV4_SM80_HC` | fallback covered by target smoke | no replacement benchmark | not run | deferred | Upstream HC fast paths rely on MHC/TileLang/DeepGEMM helpers; attention/MoE kernels remain higher priority |
| `linear_bf16_fp32_fallback` | researched, deferred | bf16-direct | no initial toggle | fallback covered by target smoke | no replacement benchmark | not run | leave torch | Optimize only if future profile shows HC/router fp32 matmul visible |

## Notes For Future Codex Runs

- Always update the R&D Completion Matrix in this file after attempting a
  kernel replacement.
- Store benchmark JSON or logs under `/tmp` and copy only the important result
  paths into the matrix.
- Do not promote fp4/fp8 approximate paths based only on microbench numbers.
  They need DSV4 forward smoke and oracle/top-k evidence.
- If `sgl_kernel`, FlashMLA, DeepGEMM, or FlashInfer is used only as a reference,
  say that explicitly in the matrix note.

## Non-Goals

- Do not require DeepGEMM at runtime for sm80.
- Do not depend on sm90/sm100-only cubins.
- Do not remove the torch fallback path.
- Do not enable any new high-performance path by default before the acceptance
  ladder is complete.

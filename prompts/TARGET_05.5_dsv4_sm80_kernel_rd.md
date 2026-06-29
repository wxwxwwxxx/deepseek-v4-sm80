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
| `dsv4_sparse_attention_two_source_bf16` | q + SWA cache + optional C4/C128 cache + per-source metadata -> output | SGLang DSV4 FlashMLA call contract for semantics only | sm80 local bf16 decode kernel that consumes compressed C4/C128 candidates and SWA window as distinct KV sources; specialize real DSV4 `head_dim=512`. | `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16=1` |
| `flash_mla_with_kvcache` | q + paged KV + metadata -> output | `sgl_kernel.flash_mla` | Treat FlashMLA as reference only unless sm80 binary support exists. Implement local sparse MLA path first. | experimental only |
| `fused_q_indexer_rope_hadamard_quant` | indexer q + weight + positions -> fp8 q + weights | `FusedQIndexerRopeHadamardQuantKernel` | Two lanes: bf16-direct indexer logits first; fp8 q quant only if bf16 bandwidth is bottleneck. | `MINISGL_DSV4_SM80_INDEXER_BF16=1`, `MINISGL_DSV4_SM80_INDEXER_FP8=1` |
| `fused_q_indexer_rope_hadamard_fp4_quant` | indexer q -> packed fp4 q + scale + weights | sm100-oriented fp4 path, DeepGEMM fp8/fp4 paged logits | Keep unsupported by default. Implement only after fp8 indexer works; require oracle gate because fp4 approximation risk is higher. | `MINISGL_DSV4_SM80_INDEXER_FP4=1` |
| `quantized_linear_ref` | x + fp8/fp4 weight + scale -> linear | DeepGEMM, FlashInfer, Marlin-like GEMM paths | First target MoE large shapes. Avoid naive small-shape TileLang FP4 as default. Prefer grouped route-aware fp4 dequant + MMA. | `MINISGL_DSV4_SM80_FP4_GEMM=1`, `MINISGL_DSV4_SM80_FP8_GEMM=1` |
| `wo_a_grouped_projection_fallback` | `o[tokens, groups, d]`, fp8 weight -> LoRA output | DeepGEMM `fp8_einsum` | First bf16 dequant-on-load Triton einsum. Then optional fp8 activation quant + grouped MMA. | `MINISGL_DSV4_SM80_WO_A_BF16=1`, `MINISGL_DSV4_SM80_WO_A_FP8=1` |
| `silu_and_mul_clamp_fallback` | gate/up activations -> SwiGLU output | `silu_and_mul_clamp` | Triton elementwise fusion. This is a low-risk early win. | `MINISGL_DSV4_SM80_SWIGLU=1` |
| `silu_and_mul_*_post_quant` | expert activations -> quantized activation | SGLang MoE post-quant kernels | Implement only with grouped fp4/fp8 GEMM work; post-quant alone likely adds overhead. | tied to GEMM toggles |
| `moe_gate_fallback`, `hash_topk_fallback`, `mega_moe_pre_dispatch_fallback` | router scores/ids -> weights, indices, grouped dispatch | `hash_topk`, `mask_topk_ids`, `mega_moe_pre_dispatch` | Replace Python expert route loops with exact route-aware grouping. Do not reuse graph capture dense invalid-route fallback as a default path. | `MINISGL_DSV4_SM80_MOE_ROUTE=1` |
| `hc_pre/post/head_fallback` | HC residual mix tensors -> mixed states | MHC helper paths | Lower priority. Add Triton HC split/Sinkhorn only after attention and MoE wins land. | `MINISGL_DSV4_SM80_HC=1` |
| `linear_bf16_fp32_fallback` | bf16/fp32 linear helper | `linear_bf16_fp32` | Upstream-first opt-in: cache HC fp32 weights as bf16, then use `torch.mm(..., out_dtype=torch.float32)` on CUDA/sm80. Do not write a custom matmul unless caller profiles still justify it. | `MINISGL_DSV4_SM80_LINEAR_BF16_FP32` |

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
   - bf16 `dsv4_sparse_attention_two_source_bf16`

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
| `paged_mqa_attention_fallback` | implemented opt-in | bf16-direct flat cache Triton decode | `MINISGL_DSV4_SM80_PAGED_MQA_BF16` | passed: CUDA parity for empty/single/duplicate rows plus default/toggle target smoke, 20 passed | candidates 32/128/640/1024: 3.329/3.302/3.300/3.374 -> 0.064/0.058/0.065/0.089 ms, 37.85-57.19x | forward smoke only; perf matrix harness absent | keep opt-in; promotion candidate after real DSV4 E2E confirms decode benefit | Adds `DSV4PagedMQAMetadata` indptr/indices/lengths and local online-softmax bf16 kernel for max 1024 candidates; list path remains as fallback. Artifact `/tmp/dsv4_paged_mqa_attention_bf16_microbench_20260629.json` |
| `dsv4_sparse_attention_two_source_bf16` | implemented opt-in, NCU basic tuned | bf16-direct two-source local CUDA decode | `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16` | passed: sm80 CUDA wrapper parity for compressed+SWA, no-sink, SWA-only, duplicate/empty rows; backend helper proves C4 cache is consumed; wrapper tests 11 passed; attention metadata tests 5 passed | realistic h64/tokens8 candidates 128/640/1024: torch reference 5.063/5.098/5.109 ms -> local CUDA 0.141/0.688/1.021 ms, 5.00-35.81x; adapter length-scan path 0.160/0.696/1.040 ms | toggle-on DSV4 forward fallback smoke 4 passed; no full DSV4 E2E perf matrix yet | keep opt-in; required for real DSV4 sparse attention on sm80, revisit small-batch grid parallelism after E2E profiles | Adds `python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu`, wrapper, and backend dispatch for ratio 0/4/128. Artifact `/tmp/dsv4_sparse_attention_two_source_bf16_microbench_20260629.json`. Kernel uses single-pass online softmax and value accumulation over compressed plus SWA candidates. NCU h64/640 q-register tuning: duration 1.04 ms -> 710.66 us; static shared 2.09 KB/block -> 44 B/block; L1 hit 83.14% -> 92.23%. Reports `/tmp/dsv4_sparse_attention_ncu_640_h64_before.txt`, `/tmp/dsv4_sparse_attention_ncu_640_h64_after_qreg.txt`. |
| `flash_mla_with_kvcache` / `flash_mla_sparse_prefill` | not planned for sm80 | experimental/reference-only | experimental only | fallback/unsupported paths covered | not run | not run | do not implement for sm80; superseded by local two-source sparse attention | Upstream FlashMLA does not support sm80, and mini lacks the FlashMLA packed cache layout/metadata contract. Keep imports only as reference/unsupported wrappers; use `dsv4_sparse_attention_two_source_bf16` for the sm80 DSV4 sparse attention path. |
| `k_norm_rope_cache_fallback` | implemented opt-in | bf16-direct Triton flat-cache store; fp8 deferred | `MINISGL_DSV4_SM80_KV_BF16`; `MINISGL_DSV4_SM80_KV_FP8` remains unimplemented | passed: CPU wrapper parity for norm+RoPE+non-contiguous cache loc, sm80 CUDA fused parity, target smoke 20 passed, CPU forward smoke with KV toggle 3 passed | tokens 1/8/64/512: 0.800/0.814/0.804/0.783 -> 0.068/0.066/0.064/0.065 ms, 11.69-12.47x | forward smoke only; perf matrix harness absent | keep opt-in; promotion candidate after DSV4 E2E confirms bf16 cache benefit | Refactored wrapper interface to accept norm weight, norm eps, cache tensor, and output locations while preserving RoPE-only compatibility. DSV4 model now fuses CUDA norm+RoPE+SWA cache write and skips duplicate backend store under the KV toggle. Also tightened Triton RoPE range reduction/single-store fused norm+RoPE behavior for high positions. Artifact `/tmp/dsv4_k_norm_rope_cache_bf16_microbench_20260629.json` |
| `compress_forward_fallback` | implemented opt-in | bf16-direct vectorized torch | `MINISGL_DSV4_SM80_COMPRESS` | passed: ratio-4 overlap CUDA parity plus target smoke | 163.730 -> 0.207 ms, 790.76x | forward smoke only; perf matrix harness absent | keep opt-in; highest P1 candidate | Replaced Python pooling loop with batched ratio grouping; artifact `/tmp/dsv4_sm80_target05_5_microbench_20260628.json` |
| `compress_norm_rope_store_fallback` | implemented opt-in | bf16-direct Triton norm/RoPE/cache scatter | `MINISGL_DSV4_SM80_COMPRESS_STORE` | passed: CPU fallback parity for compressed norm+RoPE+store, real C4/C128/indexer cache writes, sm80 CUDA fused parity, target smoke 20 passed, COMPRESS_STORE ratio4 forward smoke passed | store-only 2.71-3.04x; norm+store 3.63-7.21x; norm+RoPE+store 7.37-7.90x | forward smoke only; perf matrix harness absent | keep opt-in; promotion candidate after DSV4 E2E confirms compressed-cache benefit | Wrapper now accepts positions, optional norm weight/eps, RoPE config, and cache_type for compressed/indexer caches. Model delays compressor norm under the toggle so the fused store owns norm+RoPE+scatter. Artifact `/tmp/dsv4_compress_norm_rope_store_bf16_microbench_20260629.json` |
| `topk_transform_512_fallback` / v2 | implemented opt-in full upstream-first, vendored | score -> raw/page/full top-k via local CUDA v1 JIT borrowed from SGLang; torch full fallback; old indices padding retained | `MINISGL_DSV4_SM80_TOPK` | passed: CPU full-transform parity, sm80 local-CUDA-or-fallback parity, target smoke 21 passed; local CUDA v1 backend observed | B/L 1/1024, 4/2048, 16/4096: torch 0.617/0.640/0.641 ms -> local CUDA v1 0.142/0.141/0.142 ms, 4.34-4.55x | forward smoke only; perf matrix harness absent | keep opt-in; full semantics ready for bf16 indexer logits; v2 remains blocked | Adds `DSV4TopKTransformOutput` with raw/page/full indices. CUDA v1 code is vendored in `python/minisgl/kernel/csrc/jit/dsv4_topk_v1.cu` and marked `It's borrowed from SGLang`, so runtime no longer depends on full `sglang` or `/workspace/sglang-main`. Uses mini's existing `apache-tvm-ffi` JIT. Installed `sgl_kernel` lacks `deepseek_v4_topk_transform_512`; upstream v2 fails locally on PTX mbarrier/cp_async_bulk compile. Artifact `/tmp/dsv4_topk_transform_full_local_cuda_v1_microbench_20260629.json` |
| `fused_q_indexer_rope_hadamard_quant` | implemented opt-in | bf16-direct query/cache Hadamard + Triton bf16 logits + full top-k; fp8 deferred | `MINISGL_DSV4_SM80_INDEXER_BF16`; `MINISGL_DSV4_SM80_INDEXER_FP8` remains unimplemented | passed: CPU RoPE+Hadamard/logits/top-k parity, backend metadata update, sm80 Triton logits parity, target smoke 24 passed, ratio4 forward with indexer toggle passed | B/L 1/1024, 4/2048, 16/4096: torch+torch 1.416/3.467/11.476 ms -> triton+local_cuda_v1 0.299/0.289/0.359 ms, 4.73-32.00x | forward smoke only; perf matrix harness absent | keep opt-in; bf16 structure is ready, fp8 stays deferred until an fp8 logits consumer is useful on sm80 | Adds `DSV4IndexerSelectOutput`, bf16 query RoPE+Hadamard, indexer KV Hadamard-on-store, Triton paged bf16 logits, and metadata update of `c4_sparse_{raw,page,full}_indices`. Artifact `/tmp/dsv4_indexer_bf16_microbench_20260629.json` |
| `fused_q_indexer_rope_hadamard_fp4_quant` | blocked | fp4-act | `MINISGL_DSV4_SM80_INDEXER_FP4` | unsupported path test passes clear error | not run | not run | keep disabled | fp4 path is sm100/DeepGEMM-oriented; DeepGEMM fails to load locally with missing `libcudart.so.13`; requires oracle/top-k gate before any retry |
| `quantized_linear_ref` | implemented opt-in, decode-gated mixed perf | bf16 activation plus fp8/fp4 weight dequant-on-load Triton | `MINISGL_DSV4_SM80_FP4_GEMM`, `MINISGL_DSV4_SM80_FP8_GEMM` | passed: CUDA wrapper parity and benchmark allclose after fallback gates; wrapper tests 6 passed | fp8 dense m<=16: 0.62-0.82x; fp4 w13 m<=8: 0.87-0.90x; fp4 w2 m<=8: 1.50x; larger m falls back | not run; per-kernel microbench only | keep disabled by default; not a global promotion candidate, but fp4 w2 decode path is useful evidence for grouped MoE work | `python/minisgl/kernel/triton/deepseek_v4.py`; `benchmark/offline/deepseek_v4_quantized_linear_microbench.py`; artifact `/tmp/dsv4_quantized_linear_ref_bf16_weight_dequant_microbench_20260628.json`; DeepGEMM still fails to load on local CUDA 12.8 env due missing `libcudart.so.13`, FlashInfer low-precision GEMMs do not match bf16-act weight-dequant semantics |
| `wo_a_grouped_projection_fallback` | implemented opt-in, decode-gated | bf16-direct Triton fp8 weight dequant-on-load | `MINISGL_DSV4_SM80_WO_A_BF16`; `MINISGL_DSV4_SM80_WO_A_FP8` remains unimplemented | passed: CPU fallback shape/dtype, CUDA parity, default/all-toggle target smoke 20 passed | DSV4 dims G=8/R=1024/D=4096: tokens 1/8 0.761 -> 0.627 ms, 1.21x; tokens 64/512 gated to fallback, 1.00-1.09x; allclose_4e_2 | forward smoke only; perf matrix harness absent | keep opt-in and decode-gated; not a default promotion candidate until E2E confirms decode benefit | Upstream DeepGEMM `fp8_einsum` still fails to load due missing `libcudart.so.13`; installed `sgl_kernel` has no fp8_einsum equivalent. Local Triton avoids full weight materialization for tokens <=16; larger token counts fall back because the naive dequant-on-load path rereads weights per token tile. Artifact `/tmp/dsv4_wo_a_grouped_projection_bf16_weight_dequant_microbench_20260628.json` |
| `silu_and_mul_*_post_quant` | blocked | fp8/fp4 post-quant | tied to GEMM toggles | unsupported paths raise clear errors | not run | not run | keep disabled | Post-quant alone would add overhead without a fused grouped GEMM consumer |
| `moe_gate/hash_topk/mega_moe_pre_dispatch` | implemented opt-in | bf16 route grouping + grouped fp4 weight dequant-on-load Triton | `MINISGL_DSV4_SM80_MOE_ROUTE` | passed: route metadata CPU parity, CUDA routed expert parity, default/all-toggle target smoke 20 passed | current fallback -> grouped: 4.53x `decode_tiny`, 22.65x `decode_grouped`, 22.13x `prefill_grouped`; grouped vs bf16 oracle max_abs 0.0/0.0/0.25 | forward smoke only; perf matrix harness absent | keep opt-in; not a default promotion candidate until E2E/oracle gates cover bf16-direct activation semantics | Adds local route plan and grouped fp4 MoE consumer without Python expert loop in compute path; current quant-act fallback differs from bf16-direct on larger hidden shapes by design; artifact `/tmp/dsv4_moe_route_dispatch_bf16_grouped_microbench_20260628.json` |
| `hc_pre/post/head_fallback` | deferred after research | bf16-direct | `MINISGL_DSV4_SM80_HC` | fallback covered by target smoke | no replacement benchmark | not run | deferred | Upstream HC fast paths rely on MHC/TileLang/DeepGEMM helpers; attention/MoE kernels remain higher priority |
| `linear_bf16_fp32_fallback` | implemented opt-in upstream-first | bf16-direct cached HC weight copy + cuBLAS/PyTorch `torch.mm` fp32 output | `MINISGL_DSV4_SM80_LINEAR_BF16_FP32` | passed: sm80 CUDA wrapper direct path, fp32-weight fallback gate, HC head bf16-cache invalidation, all-toggle wrapper test | HC real K/N, M 1/8/128/2048: HC pre 0.0399/0.0390/0.0401/0.8474 -> 0.0357/0.0367/0.0377/0.0592 ms, 1.06-14.32x; HC head 0.0370/0.0380/0.0397/0.3869 -> 0.0344/0.0375/0.0395/0.0678 ms, 1.00-5.71x | not run; per-kernel microbench only | keep opt-in; useful for large prefill HC, not a decode-small-M default promotion yet; requires caller-level correctness gate because fp32 HC weights are rounded to bf16 cache | Adds `benchmark/offline/deepseek_v4_linear_bf16_fp32_microbench.py`; HC caches are lazy and versioned in `models/deepseek_v4.py`; artifact `/tmp/dsv4_linear_bf16_fp32_upstream_microbench_20260629.json`; max_abs vs current fp32-weight fallback 0.260-0.936, mean_abs 0.093-0.173 for random HC shapes; error vs bf16-weight fallback <=0.002 |
| `sm80_v0_bf16_bundle_e2e` | completed smoke | bf16-direct bundle | `MINISGL_DSV4_SM80_V0_BF16` | passed: bundle env policy, whitelist-only activation, excluded toggles stay disabled, wrapper bundle parity, and TARGET 05.7 correctness suite 30 passed | not run; bundle smoke only | passed: `/models/DeepSeek-V4-Flash` fallback and v0_bf16 generated 4 tokens with prompt_len 16, decode_len 4, batch 1 on A100 sm80 TP=4 | keep opt-in as TARGET 06 baseline; not promoted default | Artifacts `/tmp/dsv4_v0_fallback_smoke.json`, `/tmp/dsv4_v0_bf16_smoke.json`, plus `.rank0`-`.rank3` JSON files. The 149G checkpoint OOMs on TP=1 A100-80GB, so smoke used `torchrun --standalone --nproc_per_node=4` with PyTorch/NCCL collectives (`use_pynccl=false`) and `distributed_addr=env://`. |

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

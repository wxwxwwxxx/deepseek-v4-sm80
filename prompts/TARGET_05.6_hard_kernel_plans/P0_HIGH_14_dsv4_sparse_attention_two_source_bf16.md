# P0 HIGH: DSV4 Sparse Attention Two-Source bf16 Plan

## Why This Is High Priority

The current sm80 attention replacement accelerates a flat-cache paged-MQA
fallback, but real DeepSeek V4 sparse attention needs to consume compressed
C4/C128 candidates and the SWA cache as distinct KV sources. FlashMLA is not a
viable sm80 dependency, so mini needs a local bf16 sparse attention path before
the DSV4 attention stack can be considered high-performance.

## Policy

- Target sm80 only.
- Keep q, SWA KV, C4 KV, and C128 KV in bf16 for the first implementation.
- Do not depend on FlashMLA, FlashInfer MLA, or sm90/sm100 packed-cache
  layouts.
- Preserve the existing torch fallback and the existing flat-cache paged-MQA
  opt-in path.
- Add fp8 KV packing/dequant only after the bf16 two-source path is correct and
  useful end to end.

## Current State

Mini already builds SGLang-style sparse metadata and has opt-in Triton
`paged_mqa_attention_bf16` for a single flat cache. That kernel is useful
evidence, but it is not the full DSV4 sparse attention path:

- It falls back for real DSV4 `head_dim=512`.
- It reads candidates from one flat cache tensor.
- The backend currently maps compressed candidates back to full-token locations
  and gathers from SWA cache, instead of reading C4/C128 compressed KV buffers.

## Typical Workloads

- Decode first: one active token per request, local query heads, `head_dim=512`.
- Ratio 0 layers: SWA-only attention.
- Ratio 4 layers: C4 top-k compressed candidates plus SWA window.
- Ratio 128 layers: C128 compressed candidates plus SWA window.
- Candidate counts around 128, 640, and 1024 are the first benchmark targets.

## Implementation Plan

1. Split attention metadata by source.
   - Keep SWA indices as full-token/SWA-cache locations.
   - Keep C4 and C128 indices in compressed-cache location space.
   - Store per-row lengths for compressed candidates and SWA candidates.
   - Preserve deterministic ordering and duplicate handling.

2. Add a new wrapper boundary.
   - Proposed toggle: `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16=1`.
   - Inputs should include `q`, `swa_cache`, optional `compressed_cache`,
     compressed indices/lengths, SWA indices/lengths, `softmax_scale`, and
     optional `attn_sink`.
   - Ratio 0 should use the same wrapper with an absent compressed source.

3. Implement decode kernel first.
   - Compute logits over compressed candidates and SWA candidates in one online
     softmax stream.
   - Accumulate output in fp32 and store bf16.
   - Specialize for `head_dim=512`; do not inherit the current `dim <= 256`
     restriction.
   - It is acceptable for v1 to split the value dimension into two 256-wide
     blocks if that keeps the kernel simple and correct.

4. Wire backend dispatch.
   - For ratio 4, read `kvcache.c4_cache(layer_id)` plus `swa_cache(layer_id)`.
   - For ratio 128, read `kvcache.c128_cache(layer_id)` plus
     `swa_cache(layer_id)`.
   - For ratio 0, read only `swa_cache(layer_id)`.
   - Fall back to the existing `paged_mqa_attention_fallback` when shapes,
     devices, or candidate counts are unsupported.

5. Defer prefill.
   - Use decode-style per-query sparse attention for correctness experiments
     only.
   - Design a separate sparse prefill kernel after decode E2E data exists.

## Validation

- Unit parity against a torch two-source reference for ratio 0, 4, and 128.
- CUDA parity for `head_dim=512`, duplicate candidates, empty compressed source,
  empty SWA source, and `attn_sink` on/off.
- Backend tests proving C4/C128 attention consumes compressed caches, not only
  SWA cache remapped full-token rows.
- Microbench candidate counts 128, 640, and 1024 with adapter/metadata cost
  included.
- DSV4 forward smoke with the new toggle on.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `dsv4_sparse_attention_two_source_bf16` with correctness, microbench,
decision, and artifact paths.

## 2026-06-29 Implementation Note

- Added opt-in local CUDA JIT kernel
  `python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu`.
- Added wrapper and backend dispatch behind
  `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16=1`.
- Correctness:
  `pytest -q -o addopts='' tests/kernel/test_deepseek_v4_wrappers.py`
  passed 11 tests; `tests/attention/test_deepseek_v4_backend_metadata.py`
  passed 5 tests; toggle-on
  `tests/models/test_deepseek_v4_forward_fallback.py` passed 4 tests.
- Microbench artifact:
  `/tmp/dsv4_sparse_attention_two_source_bf16_microbench_20260629.json`.
- Current v1 is decode-only and uses single-pass online softmax plus value
  accumulation. Next performance work should tune occupancy and memory behavior
  with real DSV4 E2E profiles.

## 2026-06-29 NCU Basic Tuning Note

- Added focused NCU workload
  `benchmark/offline/deepseek_v4_sparse_attention_two_source_ncu_workload.py`.
- Profiled realistic `heads=64`, `tokens=8`, `candidates=128/640/1024`
  workload because `/models/DeepSeek-V4-Flash/config.json` has
  `num_attention_heads=64`.
- Replaced per-candidate shared-memory q loads with per-thread q values kept in
  registers. For the h64/candidates640 NCU case, duration improved from
  `1.04 ms` to `710.66 us`, static shared memory dropped from
  `2.09 KB/block` to `44 B/block`, and L1/TEX hit rate improved from `83.14%`
  to `92.23%`.
- Final microbench artifact:
  `/tmp/dsv4_sparse_attention_two_source_bf16_microbench_20260629.json`.
- NCU text reports:
  `/tmp/dsv4_sparse_attention_ncu_128_h64_before.txt`,
  `/tmp/dsv4_sparse_attention_ncu_640_h64_before.txt`,
  `/tmp/dsv4_sparse_attention_ncu_1024_h64_before.txt`,
  `/tmp/dsv4_sparse_attention_ncu_640_h64_after_qreg.txt`.
- Remaining NCU signal: h64/tokens8 still has only `0.59` waves/SM and about
  `56%` achieved occupancy, so deeper tuning should focus on small-batch
  parallelism after real E2E decode batch profiles are available.

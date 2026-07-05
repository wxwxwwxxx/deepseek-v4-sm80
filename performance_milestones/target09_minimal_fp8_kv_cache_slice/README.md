# TARGET 09.4 DSV4 SM80 Minimal FP8 KV/Cache Slice

Date: 2026-07-05

Scope: SWA packed MLA cache store plus selected-row gather/dequant parity harness only. This milestone does not implement full FP8 KV/cache E2E, does not migrate C4/C128 compressed KV, does not replace the C4 indexer cache, does not quantize compression state, and does not migrate retained prefix pages.

## Conclusion Summary

Decision: **conditional GO to a narrow TARGET 09.5 SWA-only opt-in E2E**, not a broad FP8 KV/cache migration.

What passed:

- Source-aligned packed MLA layout: `448` noPE FP8 bytes, `64` RoPE BF16 dims, `8` UE8M0 scale/pad bytes per token, and SGLang/vLLM-style padded page stride.
- Correctness parity: random BF16 rows, real mini-style SWA out_locs/page tables/SWA selected indices, tail-heavy decode, and prefix-hit/remap touched-row cases all passed.
- RoPE tail stayed BF16 and matched exactly.
- Selected-row gather/dequant was used; no full-cache dequant path was used as a performance candidate.
- CUDA graph capture/replay passed for bs buckets `1,2,4,8,16` with preallocated output workspace and zero replay allocation delta.
- Capacity ledger matched TARGET 09.3: SWA-only persistent saving was **0.576080 GiB/rank**.

What did not yet justify promotion:

- The standalone slice is slower than the current BF16 boundary: FP8 packed store plus selected gather/dequant costs about `0.060-0.062 ms` versus BF16 store plus gather at about `0.043-0.045 ms` for the measured buckets. This is acceptable for proving the boundary, but it is not acceptable as a final E2E design if each layer pays the same separated-kernel overhead.

Therefore 09.5 should continue only if it keeps the scope narrow and fuses or embeds the source-aligned store/gather work into the real mini attention path. Do not expand to C4/C128/indexer until SWA-only opt-in macro runs show acceptable latency.

## Artifacts

Harness:

```text
performance_milestones/target09_minimal_fp8_kv_cache_slice/scripts/swa_packed_mla_slice_harness.py
```

Result JSON:

```text
performance_milestones/target09_minimal_fp8_kv_cache_slice/summaries/swa_packed_mla_slice_harness.json
```

Command:

```bash
python performance_milestones/target09_minimal_fp8_kv_cache_slice/scripts/swa_packed_mla_slice_harness.py \
  --warmup 20 \
  --iters 200 \
  --graph-replays 5 \
  --bench-buckets 1,2,4,8,16 \
  --graph-buckets 1,2,4,8,16 \
  --output performance_milestones/target09_minimal_fp8_kv_cache_slice/summaries/swa_packed_mla_slice_harness.json
```

Environment:

- GPU: NVIDIA A100-SXM4-80GB
- CUDA capability: sm80
- Torch: `2.9.1+cu128`

## Implementation / Port Scheme

Implemented a standalone, off-by-default harness. It does not alter mini runtime defaults.

The harness implements:

- BF16 baseline store: flat `swa_cache[loc, 512] = row`.
- BF16 baseline selected gather: gather only selected flat rows.
- FP8 packed store: BF16 row to `448` E4M3FN noPE bytes plus `64` BF16 RoPE dims plus `8` UE8M0 scale/pad bytes.
- FP8 selected-row gather/dequant: arbitrary flat SWA row indices to a preallocated BF16 workspace.
- PyTorch byte oracle for packed store, using `torch.float8_e4m3fn`.
- CUDA graph capture/replay check over preallocated cache, loc, index, row, and output tensors.

The Triton store/gather kernels use software E4M3FN encode/decode. This follows mini's existing SM80 FP8 indexer style and vLLM's SM80 fallback behavior, avoiding a Hopper-only native FP8 assumption.

## SGLang/vLLM Layout Parity

| Surface | Source behavior | Harness behavior | Result |
| --- | --- | --- | --- |
| SGLang packed SWA store | `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py` defines `_MLA_NOPE_DIM=448`, `_MLA_SLOT_BYTES=576`, `_MLA_SCALES_PER_TOKEN=8`, RoPE copied as BF16, scales after token data. | Same constants and region order. Store writes noPE bytes, BF16 RoPE bytes, and UE8M0 scale bytes. | Matched. |
| SGLang page stride | `DeepSeekV4SingleKVPool.create_buffer` pads `page_size * 584` to a multiple of `576`. | `page_size=256` gives `149,760 B/page`: `256*576` token data, `256*8` scale bytes, `256` pad bytes. | Matched. |
| vLLM packed scatter | `fused_compress_quant_cache.py` scatters FP8 noPE, BF16 RoPE bytes, and UE8M0 scales into paged uint8 cache. | Same page/block/slot offset formula. | Matched. |
| vLLM selected gather/dequant | `cache_utils.py` gathers selected rows and dequants noPE by UE8M0 scale without full-cache dequant. | One Triton program per selected flat row; invalid rows become zeros. | Matched. |
| mini SWA boundary | `DeepSeekV4KVCache.store_swa` stores BF16 rows at `out_loc`; `DSV4AttentionBackend` uses `metadata.swa_page_indices` as selected flat cache locs. | Harness generates mini-style token page tables, `out_locs`, physical page tables, and `swa_indices`; store/gather use those flat locs. | Matched for the SWA slice. |

## Correctness Parity

Tolerance: noPE max abs error `<= 0.25`, noPE mean abs error `<= 0.02`, RoPE max abs error `== 0`. The noPE tolerance is for E4M3FN plus power-of-two UE8M0 scale error at this cache boundary; it is not a model-quality acceptance threshold.

| case | stored rows | valid selected rows | noPE max | noPE mean | RoPE max | byte mismatch | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_bf16_rows | 225 | 128 | 0.218750 | 0.018004 | 0.0 | 0 | yes |
| real_mini_decode_bs16 | 2048 | 2048 | 0.250000 | 0.017983 | 0.0 | 0 | yes |
| tail_heavy_decode_bs16 | 2048 | 2048 | 0.250000 | 0.017971 | 0.0 | 0 | yes |
| prefix_hit_remap_touched_rows_bs16 | 320 | 2048 | 0.250000 | 0.017941 | 0.0 | 0 | yes |

Notes:

- `byte mismatch = 0` means the Triton packed store matched the PyTorch FP8 byte oracle for the packed cache.
- The prefix-hit/remap case shares retained prefix pages across requests and verifies only SWA rows touched by the slice. C4/C128/indexer/state are intentionally out of scope.
- RoPE tail was copied and gathered as BF16 bytes; it stayed exact in all cases.

## Microbench

Times are milliseconds per operation, CUDA event timed, `warmup=20`, `iters=200`. Shapes are DSV4 SWA decode shapes: `head_dim=512`, `window_size=128`, `page_size=256`, `num_pages=128`.

| bs bucket | selected rows | BF16 store | BF16 gather | BF16 combined | FP8 store/quant | FP8 gather/dequant | FP8 combined | FP8 combined delta | workspace |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.021659 | 0.021668 | 0.044858 | 0.030063 | 0.028941 | 0.060692 | +0.015833 | 0.12 MiB |
| 2 | 256 | 0.021514 | 0.021029 | 0.042947 | 0.029673 | 0.028637 | 0.060492 | +0.017545 | 0.25 MiB |
| 4 | 512 | 0.021017 | 0.021151 | 0.044360 | 0.029809 | 0.028838 | 0.061223 | +0.016863 | 0.50 MiB |
| 8 | 1024 | 0.021147 | 0.021143 | 0.044981 | 0.029707 | 0.028556 | 0.061709 | +0.016728 | 1.00 MiB |
| 16 | 2048 | 0.021798 | 0.021380 | 0.045042 | 0.029286 | 0.028287 | 0.061205 | +0.016163 | 2.00 MiB |

HBM traffic estimates count row reads/writes and selected output writes; they do not claim achieved bandwidth because these kernels are launch/latency dominated at these sizes.

| bs | BF16 combined traffic | FP8 combined traffic | BF16 effective GB/s | FP8 effective GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.252 MiB | 0.198 MiB | 5.89 | 3.42 |
| 2 | 0.504 MiB | 0.395 MiB | 12.30 | 6.85 |
| 4 | 1.008 MiB | 0.791 MiB | 23.82 | 13.54 |
| 8 | 2.016 MiB | 1.582 MiB | 46.99 | 26.88 |
| 16 | 4.031 MiB | 3.163 MiB | 93.85 | 54.19 |

Interpretation:

- Persistent cache traffic is lower for FP8 packed rows, but the standalone software encode/decode kernels are slower than the BF16 baseline at these buckets.
- The result argues against a separated production path.
- The result still supports a narrow 09.5 because the capacity win is large and the next question is whether fused store and attention-integrated selected gather/dequant can hide or reduce this overhead.

## Graph Safety

Graph capture/replay used preallocated packed cache, BF16 output workspace, row tensors, loc tensors, and selected-index tensors. Replay did not allocate.

| bs bucket | captured | replay/eager | max replay alloc delta | workspace | capture alloc delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | true | 5/0 | 0 B | 0.12 MiB | 1024 B |
| 2 | true | 5/0 | 0 B | 0.25 MiB | 1024 B |
| 4 | true | 5/0 | 0 B | 0.50 MiB | 1024 B |
| 8 | true | 5/0 | 0 B | 1.00 MiB | 1024 B |
| 16 | true | 5/0 | 0 B | 2.00 MiB | 1024 B |

Result: **PASS** for standalone graph safety. For E2E promotion, mini must allocate the selected-row BF16 workspace before graph capture for every captured bs bucket and must not rebuild it during replay.

## Capacity Ledger Update

SWA-only persistent cache at `num_layers=43`, `page_size=256`, `num_pages=128`:

| item | bytes | GiB/rank |
| --- | ---: | ---: |
| BF16 SWA cache | 1,442,840,576 | 1.343750 |
| FP8 packed SWA cache | 824,279,040 | 0.767670 |
| Saved | 618,561,536 | 0.576080 |
| UE8M0 scale/pad bytes inside FP8 packed SWA | 11,272,192 | 0.010498 |
| Page-stride padding | 1,409,024 | 0.001312 |

Measured allocation probe:

| allocation | bytes | GiB |
| --- | ---: | ---: |
| BF16 SWA tensor delta | 1,442,840,576 | 1.343750 |
| FP8 packed SWA tensor delta | 824,279,040 | 0.767670 |
| Measured saved delta | 618,561,536 | 0.576080 |

This matches TARGET 09.3. No unexplained allocator or padding discrepancy was observed. The visible overhead is the expected UE8M0 scale region plus source-aligned page padding.

## TARGET 09.5 Decision

Decision: **conditional GO**, but only for this exact narrow scope:

- SWA-only packed MLA cache opt-in E2E.
- One explicit off-by-default flag, for example `MINISGL_DSV4_SM80_SWA_FP8_MLA_CACHE=1`.
- Allocate packed SWA cache instead of BF16 SWA cache only when the flag is set.
- Keep C4/C128 compressed KV BF16.
- Keep C4 indexer cache unchanged.
- Keep compression state BF16.
- Keep retained prefix page ownership unchanged, but add a cache-layout/version guard so BF16 and FP8 retained state cannot mix silently.
- Use source-aligned fused SWA store where possible: norm/RoPE/packed quant/store should replace the BF16 SWA store boundary rather than append another store.
- Use selected-row gather/dequant into graph-preallocated BF16 workspace, or fuse dequant into the sparse attention kernel. Do not dequantize full cache pages.
- Macro gates must include text/logit smoke, prefix hit/remap smoke, graph replay zero-eager, capacity allocation delta, and latency attribution against the promoted TARGET 10 baseline.

Recommended stop gates for 09.5:

- Any full-cache dequant in the performance path.
- Any dynamic allocation during graph replay.
- Any RoPE-tail FP8 quantization.
- Any C4/C128/indexer migration before SWA-only macro evidence.
- Macro decode regression that cannot be justified by capacity-only use. A standalone-kernel overhead shape like this target's `+0.016 ms` per layer should stop default promotion unless fused kernels remove it.

## Rollback / Disable

Current milestone rollback is trivial: no mini runtime path was changed. Do not run the standalone harness.

For the proposed 09.5 path, rollback must be:

```bash
unset MINISGL_DSV4_SM80_SWA_FP8_MLA_CACHE
# or set
MINISGL_DSV4_SM80_SWA_FP8_MLA_CACHE=0
```

The default mini path remains the existing BF16 SWA cache.

# TARGET 08.54: DSV4 SM80 SWA Direct Metadata Indexing Replay Microbench

## Status

Active TARGET 08 follow-up after TARGET 08.53.

TARGET 08.53 proved that the remaining SWA-independent direct-token metadata
gap is not caused by slower GEMM, Marlin MoE, sparse attention, NCCL, memcpy,
or memset.  The stable slow class is extra captured int64/bool
metadata/indexing kernels inside decode CUDA graph replay.

This target should reduce or eliminate those extra replay kernels with a
focused captured-replay microbench and a small opt-in implementation.  Keep the
SWA independent lifecycle contract intact.

## Goal

Find and remove the replay-captured torch metadata/indexing work introduced by
SWA independent + direct token metadata.

The target should answer:

```text
Which mini code path emits the extra compare/where/fill/div/rem/index/add/mul
kernel family?
Does SGLang avoid the same work by using a different metadata lifetime,
dtype boundary, or fused kernel boundary?
Can mini generate final graph-consumed SWA metadata buffers directly, without
per-layer long/bool tensor algebra inside captured replay?
```

Primary success is measured at replay level before macro level:

- captured microbench reproduces the extra kernel family;
- a fix removes or fuses most of those kernels;
- shortest full-model replay profile confirms the reduction;
- one final serving or historical macro confirms direction.

## Starting Evidence

Read first:

```text
performance_milestones/target08_decode_forward_kernel_census_unblock/README.md
performance_milestones/target08_decode_forward_kernel_census_unblock/kernel_census_routeb_vs_swa.md
performance_milestones/target08_decode_forward_kernel_census_unblock/low_cost_slow_kernel_reproducer.md
performance_milestones/target08_decode_forward_kernel_census_unblock/small_full_model_replay_probe.md
performance_milestones/target08_decode_forward_kernel_census_unblock/final_full_inference_validation.md
performance_milestones/target08_decode_forward_kernel_census_unblock/next_operator_target.md
prompts/TARGET_08.53_dsv4_sm80_decode_forward_kernel_census_unblock.md
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Key 08.53 facts:

```text
Shortest full-model replay profile, rank0, historical_4096_128_bs4:
  replay ranges:                 15
  Route B kernel instances:      81,960
  SWA direct kernel instances:   95,505
  extra instances:               +13,545
  replay-filtered kernel delta:  +22.673 ms
  mean decode range delta:       +1.368 ms / replay

Final serving_mixed_112req_wave16 macro:
  Route B decode bucket wall:    10.962 s
  SWA direct decode bucket wall: 12.305 s
  delta:                         +1.343 s / +12.25%
  graph replay counts:           both 441
  eager decode fallback:         both 0
```

Main extra kernel families:

```text
compare_scalar_kernel<long>
BitwiseAndFunctor<bool>
where_kernel_impl(... long ...)
FillFunctor<long>
index_elementwise_kernel
div_floor_kernel_cuda(... long ...)
remainder_kernel_cuda(... long ...)
CompareEqFunctor<long>
CUDAFunctor_add<long>
MulFunctor<long>
long/int direct-copy cast kernels
```

The `+645` launch deltas are important: for the short profile, `645` is
`15 decode replays * 43 layers`, so some work is repeated per layer per replay.
The target should prioritize eliminating per-layer replay metadata algebra.

## Suspect Mini Surfaces

Inspect these first:

```text
python/minisgl/attention/deepseek_v4.py
  _make_swa_indices_direct_token_metadata
  _make_swa_indices_from_page_table
  _copy_metadata_for_replay
  _direct_index_groups_for_replay
  _direct_index_metadata_for_replay
  _sparse_attention_two_source

python/minisgl/kvcache/deepseek_v4_pool.py
  translate_full_locs_to_swa_locs

python/minisgl/kernel/deepseek_v4.py
  direct_decode_index_metadata_for_replay
  copy_decode_metadata_for_replay

python/minisgl/kernel/triton/deepseek_v4.py
  direct_decode_index_metadata_for_replay
  _direct_decode_index_metadata_for_replay_kernel
```

Known high-probability sources:

- `translate_full_locs_to_swa_locs` currently uses torch `long` div/rem,
  comparisons, gather, multiply/add, and `where`.
- `_make_swa_indices_direct_token_metadata` builds full locs, then translates
  them to SWA locs, producing temporary tensors.
- `_sparse_attention_two_source` performs per-layer `.to(int32)` and
  `clamp(max=...)` at the attention boundary.
- `_direct_index_groups_for_replay` currently disables `direct_swa` when SWA
  independent lifecycle is enabled, so SWA direct may not use the same fused
  graph-buffer path as non-independent Route B.

## SGLang References

Use SGLang as a design reference before inventing a local mechanism:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/attn.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode_indices.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/runtime.py
```

Questions to answer:

- Does SGLang generate SWA decode metadata once per forward rather than once
  per layer?
- Where does SGLang translate full locs to SWA locs?
- Are final attention/indexer metadata tensors already `int32` before layer
  replay?
- Does SGLang use a fused Triton/CUDA kernel for full-to-SWA mapping or page
  decode indices?
- Which part can be adapted cleanly to mini without rewriting the whole
  attention backend?

Do not assume SGLang's implementation is directly faster on mini; prove the
boundary and kernel count difference.

## Required Work

### 1. Three-Way Replay Attribution

Run the smallest replay profile that preserves the 08.53 signal and compare:

```text
Route B prefix baseline
SWA independent + 08.49 page-table cache
SWA independent + 08.50 direct token metadata
```

Use the short full-model shape first:

```text
historical_4096_128_bs4
decode length 16 if supported by the harness
TP8
page size 256
num pages 128
graph buckets [1,2,4,8,16]
rank-local nsys on rank0
NVTX filter prefix batch_forward:decode:
```

Report whether the extra kernel family is caused by:

- independent SWA lifecycle itself;
- page-table cache path;
- direct token metadata path;
- attention-boundary per-layer conversions;
- or some combination.

Stop if the 08.53 extra kernel family no longer reproduces.

### 2. Captured Replay Microbench

Build a low-cost captured replay microbench that reproduces the extra
metadata/indexing kernel family without a full long macro.

The microbench should isolate, in captured CUDA graph replay:

```text
full_locs -> swa_locs translation
swa_topk_lengths production / clamp
swa_page_indices final-buffer generation
attention-boundary dtype/range handling
direct graph metadata buffer generation
```

Preserve:

- bucket sizes `1,2,4,8,16`, with bs4 and bs8 emphasized;
- SWA window `128`;
- page size `256`;
- compact SWA locs versus full Route B locs;
- realistic token locality;
- invalid `-1` density;
- dummy-token behavior from the SWA lifecycle contract;
- `swa_topk_lengths`;
- dtype, stride, alignment, and final attention metadata shape.

The microbench must use captured replay.  Eager-only timing may be recorded as
debug data, but it is not enough for a fix decision.

### 3. Candidate Fixes

Try the smallest fix that removes the extra kernels.  Reasonable candidates:

1. **Fused SWA direct graph metadata kernel for independent lifecycle**

   Extend `direct_decode_index_metadata_for_replay` so SWA independent can use
   a fused path.  The kernel should consume full-token page table / positions
   plus `full_to_swa_page`, `dummy_token_start`, and `swa_dummy_page`, then
   write final `int32 swa_page_indices` and `int32 swa_topk_lengths` directly.

2. **Precompute graph-consumed SWA metadata outside per-layer replay**

   If final SWA metadata is identical for all layers in a decode replay, build
   it once into stable graph buffers and make layer attention consume those
   buffers without per-layer torch `to/clamp/where/div/rem` work.

3. **Attention-boundary fast path**

   If metadata is already CUDA `int32`, contiguous, and width-bounded, skip
   per-layer `.to(int32)` and `clamp(max=...)` in `_sparse_attention_two_source`.
   Keep debug checks outside CUDA graph capture.

4. **SGLang-aligned kernel adaptation**

   If SGLang has an existing fused full-to-SWA / paged decode indices kernel
   that matches mini's layout closely, adapt the kernel or its boundary rather
   than writing a slow local chain of torch ops.

Prefer an opt-in environment toggle first.  Suggested name:

```text
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
```

Use a different name if the implementation scope is narrower, but keep the
toggle explicit until correctness and macro gates pass.

### 4. Kernel Count And Replay Gates

After each candidate, run the microbench and short full-model replay census.

Success metrics:

- `compare/where/fill/div/rem/add/mul/index` launch deltas fall sharply;
- the `+645` per-layer-per-replay launch deltas are removed or explained;
- short full-model replay kernel instance delta drops materially from
  `+13,545`;
- replay-filtered kernel delta drops materially from `+22.673 ms / 15 replays`;
- graph replay remains captured with eager decode `0`;
- Route B behavior is unchanged.

If a candidate only moves work from many torch kernels to one fused metadata
kernel, report the new kernel's total time and launch count.  That is acceptable
if total replay time improves.

### 5. Correctness And Macro Validation

For microbench-only or instrumentation changes, run focused tests.

For any runtime logic or kernel change, run:

```text
focused attention/kvcache/kernel tests
fixed128 SWA direct text smoke
short full-model replay profile used by 08.53
one final macro pair: serving_mixed_112req_wave16 or historical_4096_1024_bs4
```

All full-model rows must show:

- pass status;
- sane text for smoke;
- graph replay healthy;
- eager decode `0`;
- no CUDA illegal memory access;
- no NCCL watchdog;
- no SWA stale metadata, negative refcount, or double free;
- capacity plan unchanged.

Do not run large benchmark matrices while iterating.  Use one final macro pair
only after the replay-level evidence improves.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/
```

Required files:

- `README.md` with final verdict;
- `three_way_replay_attribution.md`;
- `sglang_boundary_review.md`;
- `captured_replay_microbench.md`;
- `candidate_fix_design.md`;
- `kernel_count_after.md`;
- `short_full_model_replay_after.md`;
- `correctness_macro_gate.md`;
- `next_target_or_promotion.md`;
- raw scripts/logs/profiles/summaries under `raw/`.

The README must answer:

1. Which exact path emitted the extra captured metadata/indexing kernels?
2. Was the owner independent lifecycle, direct token metadata, attention
   boundary, or fused graph metadata being disabled?
3. What does SGLang do differently at this boundary?
4. Did the captured replay microbench reproduce the extra kernel family?
5. Which candidate fix was implemented, if any?
6. How much did kernel launch count and replay-filtered time improve?
7. Did one final macro confirm the direction?
8. Should SWA direct remain opt-in, be replaced by the new fused path, or be
   abandoned in favor of another SGLang-aligned route?

## Stop Conditions

Stop and report instead of broad patching if:

- the 08.53 extra kernel family cannot be reproduced;
- three-way attribution shows the owner is not SWA metadata/indexing;
- captured microbench cannot reproduce the replay kernel family;
- SGLang uses an incompatible mechanism that would require a broad backend
  rewrite;
- a candidate fix changes SWA lifecycle correctness or prefix ownership;
- graph replay falls back to eager;
- a fix would require rewriting sparse attention kernels before the metadata
  owner is eliminated.


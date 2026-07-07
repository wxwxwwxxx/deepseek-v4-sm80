# TARGET 08.55: DSV4 SM80 Compressed Metadata Boundary Replay Cleanup

## Status

Active TARGET 08 follow-up after TARGET 08.54.

TARGET 08.54 solved the main SWA direct replay gap found by TARGET 08.53.  The
old SWA direct extra-kernel family was reduced from a full-model short replay
delta of `+13,545` launches / `+22.673 ms` to `-645` launches / `-1.638 ms`
versus Route B, while preserving graph replay and the SWA independent lifecycle
contract.

This target is the last planned small-kernel cleanup pass before returning to
TARGET 09 low-precision research, unless fresh evidence shows a large metadata
owner remains.

## Goal

Clean up the remaining C4/C128 compressed metadata boundary kernels that still
appear in captured decode graph replay after TARGET 08.54.

Use the TARGET 08.54 fused SWA direct path as the active SWA baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect_replaymetafused
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
```

The target should answer:

```text
Are the remaining target-family launches caused by C4/C128 compressed metadata
length/index boundaries?
Can they be removed by reusing replay-copied int32 graph buffers or a small
fused helper without changing sparse attention semantics?
Is the remaining metadata overhead now small enough that further small-kernel
cleanup should stop and TARGET 09 low-precision research should resume?
```

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/README.md
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/kernel_count_after.md
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/short_full_model_replay_after.md
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/correctness_macro_gate.md
performance_milestones/target08_swa_direct_metadata_indexing_replay_microbench/next_target_or_promotion.md
prompts/TARGET_08.54_dsv4_sm80_swa_direct_metadata_indexing_replay_microbench.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/TARGET_09_dsv4_sm80_low_precision_research.md
prompts/target.md
```

Key 08.54 result:

```text
Short full-model replay:
  Route B kernel instances:      81,960
  old SWA direct instances:      95,505
  08.54 fused SWA instances:     81,315
  old SWA - RouteB:              +13,545 / +22.673 ms
  08.54 SWA - RouteB:            -645 / -1.638 ms

Final serving_mixed_112req_wave16:
  Route B decode bucket wall:    11.326 s
  08.54 SWA decode bucket wall:  11.581 s
  delta:                         +0.255 s
  graph replay counts:           both 441
  eager decode fallback:         both 0
```

TARGET 08.54 interpretation:

```text
The main SWA full-to-SWA/store metadata gap is removed.
Remaining target-family launches likely come from compressed C4/C128 metadata
boundaries that still use torch long/bool expressions such as clamp/cast or
(indices >= 0).sum(...).
```

## Suspect Surfaces

Inspect these first:

```text
python/minisgl/attention/deepseek_v4.py
  _sparse_attention_two_source
  attention-boundary c4/c128 index and length handling
  c4_sparse_topk_lengths / c128_topk_lengths_clamp1 use
  _direct_index_metadata_for_replay
  _copy_metadata_for_replay

python/minisgl/kernel/deepseek_v4.py
  direct_decode_index_metadata_for_replay
  copy_decode_metadata_for_replay

python/minisgl/kernel/triton/deepseek_v4.py
  _direct_decode_index_metadata_for_replay_kernel
```

Likely owners:

- per-layer `.to(torch.int32)` or `clamp(max=...)` around C4/C128 length
  tensors;
- fallback compressed length recomputation such as `(compressed_indices >= 0)
  .sum(dim=-1)`;
- C4/C128 index buffers not using already replay-copied direct graph metadata;
- missing fast path for metadata that is already CUDA `int32`, contiguous, and
  width-bounded.

## SGLang References

Review SGLang only for boundary design, not broad backend replacement:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/attn.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode_indices.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/runtime.py
```

Questions:

- Are C4/C128 lengths and indices final `int32` graph-consumed buffers before
  layer attention?
- Does SGLang recompute compressed lengths per layer, or reuse prepared
  metadata?
- Is there a small fused paged-index helper that can be adapted to mini's
  metadata layout?

## Required Work

### 1. Replay Census Reset

Use the 08.54 fused SWA variant and rerun the shortest replay census:

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

Compare:

```text
Route B prefix baseline
08.54 SWA replaymetafused
```

Report:

- remaining kernel-family deltas;
- per-layer-per-replay launch patterns;
- whether C4, C128, or both are responsible;
- replay-filtered kernel time delta;
- graph replay count and eager decode count.

### 2. Captured Replay Microbench

Build or extend a low-cost captured replay microbench for compressed metadata
boundaries.

The microbench should isolate:

```text
C4 sparse topk length handling
C4 sparse page/raw/full index handling
C128 length handling
C128 page/raw/full index handling
attention-boundary dtype/range fast path
fallback length recomputation from indices
```

Preserve:

- bucket sizes `1,2,4,8,16`, with bs4 and bs8 emphasized;
- page size `256`;
- C4 topk `512`;
- realistic C4/C128 length distributions;
- invalid `-1` density;
- component loc ownership;
- dtype, stride, alignment, and final graph metadata shape;
- captured CUDA graph replay as the primary evidence.

### 3. Candidate Fixes

Try at most two scoped fixes.  Reasonable candidates:

1. **Compressed attention-boundary fast path**

   If C4/C128 indices and lengths are already CUDA `int32`, contiguous, and
   width-bounded, pass them directly to sparse attention and skip per-layer
   `.to(int32)`, `clamp`, and length recomputation.

2. **Replay-copied compressed length buffers**

   Ensure `c4_sparse_topk_lengths` and `c128_topk_lengths_clamp1` are final
   graph-consumed `int32` buffers, copied once per replay and reused by every
   layer.

3. **Fused compressed direct metadata helper**

   If C4/C128 direct graph metadata still requires multiple torch ops, extend
   the existing Triton helper to write final index and length buffers in one
   launch.

4. **SGLang-aligned small helper**

   If SGLang has a clean small paged-index helper matching mini's layout,
   adapt that boundary instead of hand-rolling multiple torch expressions.

Keep any runtime behavior opt-in until gates pass.  Reuse the existing 08.54
opt-in stack if the fix is naturally part of replay metadata fusion; otherwise
add a narrow toggle with a clear name.

### 4. Stop-Line / Transition Decision

This target must decide whether to stop small-kernel cleanup and return to
TARGET 09 low-precision research.

Stop further metadata small-kernel work if any of these are true:

- short replay delta versus Route B is within roughly `0.5 ms/replay`;
- final macro decode bucket gap is within roughly `2-3%` and not stable across
  repeat pairs;
- the remaining launch deltas are not per-layer-per-replay or are dominated by
  already-fused helper launches;
- a scoped fix improves replay time by less than `1%` or macro throughput by
  less than `1-2%`;
- the remaining owner is no longer metadata/indexing but compute, cache
  bandwidth, communication, or precision.

If the stop line is reached, write an explicit recommendation:

```text
Stop TARGET 08 small-kernel cleanup.
Use the best SWA opt-in stack as the prefix/SWA capacity baseline.
Resume TARGET 09 low-precision research, prioritizing INT8 MoE feasibility and
FP8/KV-cache ROI only where memory ledger still supports it.
```

Do not spend another target chasing tiny PyTorch elementwise kernels unless a
fresh profile shows they again dominate decode replay.

### 5. Validation Gates

For microbench-only or instrumentation changes, run focused tests.

For runtime logic changes, run:

```text
focused attention/kernel tests
fixed128 SWA replaymetafused text smoke
short full-model replay census
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

Do not run broad benchmark matrices while iterating.  Use one final macro pair
after replay-level evidence improves.

## Deliverables

Write results under:

```text
performance_milestones/target08_compressed_metadata_boundary_replay_cleanup/
```

Required files:

- `README.md` with final verdict;
- `replay_census_reset.md`;
- `sglang_compressed_boundary_review.md`;
- `captured_compressed_metadata_microbench.md`;
- `candidate_fix_design.md`;
- `kernel_count_after.md`;
- `short_full_model_replay_after.md`;
- `correctness_macro_gate.md`;
- `stop_line_low_precision_transition.md`;
- raw scripts/logs/profiles/summaries under `raw/`.

The README must answer:

1. Which remaining kernels were actually owned by C4/C128 compressed metadata?
2. Did the captured microbench reproduce them?
3. What does SGLang do differently at this boundary?
4. Which fix was implemented, if any?
5. How much did launch count and replay-filtered time improve?
6. Did one macro pair confirm the direction?
7. Are metadata small-kernel costs now below the stop line?
8. Should the project continue TARGET 08 cleanup, or move back to TARGET 09
   low-precision research?

## Stop Conditions

Stop and report instead of broad patching if:

- the remaining kernels are not reproducible in captured replay;
- C4/C128 compressed metadata is not the owner;
- a candidate fix would require rewriting sparse attention compute kernels;
- a candidate fix weakens component loc ownership or SWA lifecycle correctness;
- graph replay falls back to eager;
- improvement is below the stop-line thresholds above.


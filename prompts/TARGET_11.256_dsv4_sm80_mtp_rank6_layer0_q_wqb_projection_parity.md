# TARGET 11.256: DSV4 SM80 MTP rank6 layer0 q_wqb projection parity

## Status

Next after TARGET 11.255.

TARGET 11.255 classified the first true layer0 producer mismatch as:

```text
classification: layer0_attention_qkv_owner
subowner: rank6.layer0.q_wqb_output
anchor: uid0 / position5 / full_loc3077 / depth0 correction row
```

Important conclusion from 11.255:

```text
layer0.input is exact;
layer0.attention_input is exact;
layer0.wqa_output is exact;
layer0.q_lora_after_norm is exact;
rank6 layer0.q_wqb_output is the first rank-local mismatch;
wo_b all-reduce only propagates the rank6 local drift to other ranks.
```

This target must look inside the `q_wqb` projection/operator path.  Do not
reopen SWA accepted commit, C4/C128 state, layer1 store, MoE, logits, sampler,
graph, communication, or low-precision performance work unless this target
disproves the `q_wqb` owner.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Debug Harness Policy

Reusable MTP debug harnesses live under:

```text
debug/mtp/
```

Use the tracked harnesses first:

```text
debug/mtp/run_matrix.py
debug/mtp/analyze_state_parity.py
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
Milestone directories should contain reports, raw outputs, and one-off
artifacts.  If this target needs a reusable q_wqb/operator helper, put it under
`debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/
```

## Goal

Explain and, if source-aligned and low-risk, fix the rank6 layer0
`q_wqb_output` mismatch for the carried full-schedule anchor:

```text
uid0 / position5 / full_loc3077 / depth0 correction row
rank6
layer0.q_wqb_output
```

The target passes with one of these classifications:

1. `q_wqb_input_owner`: direct q_wqb input comparison is not actually exact
   once the correct rank6 row is captured.
2. `q_wqb_weight_cache_owner`: the cached/dequantized/pretransposed weight used
   by normal decode and MTP target verify differs, is stale, or is addressed
   differently.
3. `q_wqb_dispatch_owner`: normal decode and target verify dispatch different
   projection implementations or different fast-path flags.
4. `q_wqb_row_shape_owner`: the same implementation is shape-sensitive; the
   target-verify row shape produces a different result from a normal-decode
   compatible row shape.
5. `q_wqb_row_invariant_local_owner`: Mini's
   `row_invariant_local=is_target_verify` q_wqb path changes semantics or
   hides a shape-sensitive path.
6. `q_wqb_precision_backend_owner`: the mismatch is caused by FP8/BF16/Marlin
   precision or accumulation differences that are not exact under target-verify
   shape.
7. `q_wqb_rank_weight_owner`: rank6 has a rank-local q_wqb weight/cache/layout
   difference not present on other ranks.
8. `q_wqb_not_owner`: a stricter operator trace disproves 11.255's boundary and
   names the earlier true owner.
9. `q_wqb_instrumentation_no_go`: current hooks cannot compare the operator
   enough; add the smallest missing instrumentation or write a narrower target.

If a minimal source-aligned fix is clear, it may be attempted.  Otherwise stop
with the precise owner and next repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer0_output_subboundary_parity/README.md
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/README.md
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/
prompts/TARGET_11.255_dsv4_sm80_mtp_layer0_output_subboundary_parity.md
prompts/TARGET_11.254_dsv4_sm80_mtp_layer1_swa_producer_boundary_parity.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Do not restore fail-closed accepted commit.
Do not branch on rank, uid, position, layer, loc, bs, request id, token, or
prompt text.
```

Relevant 11.255 evidence:

```text
rank6 layer0.q_wqb_output baseline raw sha: e5b16d94ad3eb9ac
rank6 layer0.q_wqb_output MTP raw sha:      03e7d5ceaeafc350
baseline abs_sum: 95.66873168945312
MTP abs_sum:      95.66879272460938
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

Likely Mini code regions:

```text
DeepseekV4Attention._q_wqb_bf16_weight_cache_name
DeepseekV4Attention._q_wqb_marlin_weight_cache_name
DeepseekV4Attention.prepare_q_wqb_bf16_weight_cache
DeepseekV4Attention.prepare_q_wqb_marlin_weight_cache
DeepseekV4Attention._q_wqb_per_row_probe
DeepseekV4Attention.forward q_wqb block
Quantized linear forward / forward_fp8_cached_bf16_weight /
forward_fp8_marlin_weight
```

Current q_wqb dispatch to audit:

```text
dense_fp8_marlin_projection_enabled()
DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE
MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM
row_invariant_local=is_target_verify
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang as a contract reference for target-verify row construction and
projection dispatch when source behavior is clear.  Do not copy large code
blindly; first identify whether Mini's mismatch is dispatch, shape, cache, or
precision related.

## Non-Goals

- Do not change SWA accepted commit/copy/restore.
- Do not change C4/C128 state lifecycle.
- Do not change MoE microbatching unless a guard proves it regressed.
- Do not change attention backend, q_norm_rope, wo_a, wo_b, or all-reduce
  unless q_wqb is disproven.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision
  research.

## Work Plan

### 1. Reproduce The q_wqb Owner

Use the full `1/2/4/5/6` schedule and the carried anchor.  Confirm that the
current run still reaches:

```text
first rank-local mismatch: rank6 layer0.q_wqb_output
```

Do not use isolated bs6 as the primary decision artifact; it does not reproduce
the full-schedule allocator loc `3077`.

### 2. Capture q_wqb Operator Inputs Precisely

For rank6 layer0 and the carried row, compare:

```text
q_wqb input / q_lora_after_norm
input dtype, shape, stride, contiguous flag
input row index within the local batch
positions and target-verify row metadata
output dtype, shape, stride
```

If direct q_wqb input is not exact, classify `q_wqb_input_owner` and stop before
patching the projection operator.

### 3. Compare Dispatch And Weight Cache

For baseline normal decode and MTP target verify, record:

```text
q_wqb path string
dense_fp8_marlin_projection_enabled
DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE
MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM
row_invariant_local
cache name
owner label
weight/cache tensor dtype, shape, stride, pointer if useful
weight/checksum or stable sampled checksum
rank-local cache preparation report for rank6
```

Classify:

- `q_wqb_dispatch_owner` if the two modes dispatch different implementations
  without a source-aligned reason.
- `q_wqb_weight_cache_owner` if cache values/layout/addressing differ.
- `q_wqb_rank_weight_owner` if rank6 differs from other ranks in a way that
  explains the local-only first mismatch.

### 4. Run Row-Shape And Per-Row Oracles

Use or extend `_q_wqb_per_row_probe` for a focused operator comparison.

Compare at least:

```text
batched target-verify q_wqb output
same input rows executed one row at a time through the same q_wqb path
normal-decode compatible shape oracle if available
reference quantized_linear path if available and affordable for only layer0/rank6
cached BF16 path with row_invariant_local disabled/enabled, if applicable
```

The important question is whether target-verify's row shape changes the q_wqb
result even when input and weight are identical.

Classify:

- `q_wqb_row_shape_owner` if batched target-verify shape differs but per-row or
  normal-shape oracle matches baseline.
- `q_wqb_row_invariant_local_owner` if `row_invariant_local=is_target_verify`
  is the semantic difference.
- `q_wqb_precision_backend_owner` if only a specific precision/backend path
  drifts.

### 5. Source-Aligned Minimal Fix

If the owner is clear and the fix is small, prefer a correctness-safe path:

```text
make target-verify q_wqb use the same semantic projection as normal decode;
or microbatch/reshape only q_wqb target-verify rows into a normal-shape-safe
path;
or disable the shape-sensitive q_wqb fast path for target verify only;
or align cache preparation/read semantics with normal decode/SGLang.
```

Any fix must be generic.  Do not branch on rank6, layer0, the anchor loc, bs6,
or the prompt.

If the correctness-safe fix costs performance, leave it opt-in or clearly mark
it as a correctness gate.  Greedy exactness comes before MTP performance.

### 6. Validation

If runtime code changes, validate:

```text
full 1/2/4/5/6 MTP exactness matrix
text sanity smoke
TARGET 11.246 MoE microbatch guard
TARGET 11.247 accepted-commit guard
TARGET 11.249/11.250 C128 guards
TARGET 11.251/11.252 analyzer validity guards
TARGET 11.253/11.254/11.255 carried anchors
```

Also run static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/run_matrix.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

## Stop Conditions

Stop this target when one of these is true:

1. q_wqb input, dispatch, weight/cache, row-shape, precision/backend, or
   rank-local owner is classified with evidence.
2. A minimal source-aligned q_wqb fix lands and the exactness matrix improves
   or passes.
3. q_wqb is disproven and the earlier true owner is named.
4. Instrumentation is insufficient and the missing probe is specified exactly.

Do not continue into q_norm_rope, attention backend, wo_a/wo_b, MoE, layer1,
logits, sampler, graph, or perf after q_wqb is classified.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/README.md
```

Include:

- final classification;
- anchor and schedule used;
- rank6 q_wqb input parity table;
- dispatch/env/path table for baseline versus MTP;
- q_wqb weight/cache parity table;
- row-shape/per-row oracle results;
- SGLang source-parity notes;
- any code changes and exactness impact;
- commands and tests run;
- next target recommendation if not fully fixed.

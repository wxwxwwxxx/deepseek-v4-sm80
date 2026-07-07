# TARGET 11.14: DSV4 SM80 MTP q_wqb / q_lora Precision Boundary Parity

## Status

Next after TARGET 11.13.

TARGET 11.13 added a reusable debug-only operator parity harness and used
q_norm_rope as the pilot.  It reproduced the apparent q/RoPE owner, but showed
that q_norm_rope is a drift amplifier rather than the root owner:

```text
normal q/RoPE path        = mini.q_kv_norm_rope_cache_fallback
target-verify q/RoPE path = mini.q_kv_norm_rope_cache_fallback

q_wqb_output normal vs target:
  allclose yes at 1e-3
  not bit-exact
  max_delta = 6.1035e-05

q_after_q_norm_rope normal vs target:
  allclose no
  max_delta = 0.001953125
```

Micro probes showed that both mini standalone replay and SGLang-style torch
reference preserve the same normal-vs-target delta.  Therefore a q/RoPE-only
dispatch or SGLang fused_q_norm_rope port is not sufficient.  The next owner is
the upstream precision boundary that produces the non-bit-exact `q_wqb_output`.

## Goal

Identify and fix, or precisely no-go, the first upstream q-path boundary that
makes normal decode and MTP target-verify diverge before q_norm_rope:

```text
hidden / attention input
q_lora input
q_lora output
q_norm output
wq_b input
wq_b local output
q_wqb_output
```

The target passes when one of these is true:

1. `q_wqb_output` becomes bit-exact, or sufficiently exact to stop q/RoPE from
   becoming the first owner, followed by a rerun of the q/RoPE and exactness
   gates.
2. Or the q path is precisely no-go with a source-parity-backed plan to port a
   larger SGLang-equivalent q projection/norm/RoPE boundary.

Do not start MoE, indexer FP8, CUDA graph, throughput tuning, C128 boundary
gates, or acceptance tuning in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_operator_parity_q_norm_rope/README.md
performance_milestones/target11_mtp_rank_local_downstream_parity_census/README.md
prompts/TARGET_11.13_dsv4_sm80_mtp_operator_parity_framework_q_norm_rope_pilot.md
```

Important TARGET 11.13 result:

```text
q_norm_rope same-kernel output drift is caused by a small q_wqb_output input
perturbation.  q/RoPE is not the root dispatch owner.
```

The current priority remains:

```text
P0: q path precision boundary
P1: MoE output
P2: indexer FP8 query
P3: later-layer attention
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
```

Relevant mini q path:

```text
q_lora projection / q_lora input
q_norm
wq_b / q_wqb output
q_norm_rope / q_kv_norm_rope_cache_fallback
DSV4Linear cached BF16 / FP8 activation quantization paths if used
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/linear.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/elementwise.py
```

Relevant SGLang behavior:

```text
q_lora = self.wq_a(hidden_states)[0]
q_lora = self.q_norm(q_lora)
q = self.wq_b(q_lora)[0]
_compute_q_b / fused_q_norm_rope
```

Use SGLang source behavior as the preferred contract when mini and SGLang
differ.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id, or
  observed token numerical branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6` as the fix.
- Do not fix MoE or indexer before the q path boundary is classified.
- Do not rely only on full-generation token mismatch; use operator parity
  records and micro allclose probes.
- Do not globally change q projection precision without documenting memory,
  speed, and correctness impact.

## Work Plan

### 1. Reproduce The q_wqb Input Perturbation

Use the same contract:

```text
TP8
/models/DeepSeek-V4-Flash
page_size=256
num_pages=16
draft_len=2
decode_len=8
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
accepted commit enabled
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
MINISGL_DSV4_MTP_OPERATOR_PARITY=1
```

Reproduce at least:

```text
bs=1 event0 rank6 q_wqb_output near-exact but not bit-exact
bs=2 event0 rank6 q_wqb_output near-exact but not bit-exact
```

If the perturbation no longer reproduces, rerun the 11.13 q/RoPE probes and
follow the new first owner.

### 2. Extend The Operator Parity Harness Upstream

Reuse the 11.13 harness and add q-path records for:

```text
attention input / hidden input to q path
wq_a input
wq_a output / q_lora
q_norm input
q_norm output
wq_b input
wq_b local output before any communication if applicable
q_wqb_output
```

For each boundary, record:

```text
normal path/kernel
target-verify path/kernel
shape
dtype
allclose
bit-exact flag if cheap
max_delta / mean_delta
first differing index
sample values
owner verdict
```

Use stricter reporting than the previous 1e-3 layer-parity tolerance.  For q
path boundaries, a value can be allclose but still be an owner if it causes
q_norm_rope to cross a bf16 rounding threshold.

### 3. Source-Parity Table For q Projection Path

Before trying many flags, write a source-parity table:

```text
concept
SGLang behavior
mini normal decode behavior
mini target-verify behavior
same/different/unknown
action
```

Cover:

- wq_a input shape and hidden row ordering;
- wq_a weight path and precision;
- q_norm eps, dtype, and reduction;
- q_norm output dtype;
- wq_b input shape and row ordering;
- wq_b weight path: FP8, cached BF16, Marlin, fallback;
- FP8 activation quantization for wq_b if used;
- local matmul accumulation dtype;
- row-batched vs per-row behavior;
- whether normal decode and target verify share the same path;
- any all-reduce or TP behavior in q path.

### 4. Micro Allclose / Same-Kernel Oracles

Try these in order:

1. Captured replay:
   - run q_norm and wq_b on captured normal and target inputs;
   - determine the first boundary that is not bit-exact.
2. Same-kernel oracle:
   - force target-verify q projection path to use the same helper/kernel as
     normal decode if dispatch differs.
3. Per-row oracle:
   - run q_norm and wq_b one row at a time for target verify if normal decode is
     one row.
4. Precision oracle:
   - use a higher-precision or reference path for q_norm/wq_b to see whether
     bf16/fp8 rounding is the owner.
5. SGLang-style oracle:
   - if SGLang's q path semantics differ materially, adapt the relevant larger
     q projection/norm/RoPE boundary under `sglang_prefill_extend`.

The key question:

```text
Is q_wqb_output non-bit-exact because the input to wq_b is already different,
because wq_b itself is row/shape-sensitive, or because a quantized/cached weight
path differs between normal decode and target verify?
```

### 5. Fix Or Precisely No-Go

Preferred fixes:

1. If a dispatch/path mismatch exists:
   - make target verify use the same q path as normal decode under
     `sglang_prefill_extend`.
2. If row-batched behavior differs:
   - add a row-invariant q path for target verify, analogous to prior `wo_b`
     row-invariant fixes.
3. If FP8 activation quantization or cached BF16 weight path is the owner:
   - make q path quantization row-invariant or use the normal-decode precision
     path for target verify.
4. If q_norm precision is the owner:
   - align q_norm dtype/reduction/order with SGLang or normal decode.
5. If fixing only q_wqb is insufficient:
   - write a precise plan for a larger SGLang-style q projection + q/RoPE
     fused boundary.

Do not fix by special-casing rank6, bs=1/2, request id, or token id.

### 6. Validate Incrementally

After any fix:

```text
q path operator parity for bs=1 event0 rank6
q path operator parity for bs=2 event0 rank6
q/RoPE operator parity for bs=1 event0 rank6
q/RoPE operator parity for bs=2 event0 rank6
bs=1 targeted trace
bs=2 targeted trace
bs=1/2/4/5/6 exactness matrix
```

If the matrix still fails, rerun enough of the operator census to identify the
next owner.  Expected queued owners are MoE output and indexer FP8.

## Success Criteria

Minimum:

```text
q_wqb perturbation is reproduced
first upstream q-path owner is identified with operator records
q projection source-parity table is written
accepted commit remains enabled
no batch/rank/request/token special branch is introduced
```

Full:

```text
q_wqb_output no longer triggers q_norm_rope drift for known rows
q/RoPE parity passes for known rows
bs=1/2/4/5/6 exact, or next owner is identified by the operator framework
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- q_wqb perturbation cannot be reproduced and the operator census must be
  refreshed;
- exactness requires rank/batch/request/token special casing;
- normal decode and target verify already share the exact same q path and the
  owner is an unavoidable tiny upstream perturbation that needs a broader
  target-verify contract decision;
- fixing the q path requires a larger SGLang q projection/RoPE port;
- q path is fixed but independent MoE/indexer owners remain.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_q_wqb_q_lora_precision_boundary/README.md
```

Include:

- reproduction of the 11.13 q_wqb perturbation;
- operator parity harness extensions;
- q projection source-parity table against SGLang;
- micro allclose / same-kernel / per-row / precision probe results;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- next owner and next focused target if needed, or TARGET 11.3 go/no-go.

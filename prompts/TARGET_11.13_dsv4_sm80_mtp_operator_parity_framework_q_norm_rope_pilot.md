# TARGET 11.13: DSV4 SM80 MTP Operator Parity Framework + q_norm_rope Pilot

## Status

Next after TARGET 11.12.

TARGET 11.12 showed that after fixing the earlier target-verify metadata,
attention/KV, `wo_a`, and `wo_b` owners, the remaining MTP correctness failures
are not one loose bug.  They are multiple rank-local downstream parity owners:

```text
P0: q_norm_rope / q_after_q_norm_rope rank-local drift
P1: MoE output drift with exact MoE input
P2: indexer FP8 query uint8 drift with exact input boundary
P3: later-layer attention drift
```

The highest-priority owner is q/RoPE because it appears in both bs=1 and bs=2,
is earliest in layer order, is rank-local before reduce, and its input is exact
within trace tolerance.

This target should create a reusable operator-parity debugging framework and use
q_norm_rope as the first pilot.

## Goal

Make target-verify operator debugging systematic:

```text
normal target decode row/operator
vs
MTP target-verify row/operator
```

for the same model, layer, rank, request, token, position, and visible prefix.

Then apply that framework to q_norm_rope:

```text
q_wqb_output -> q_after_q_norm_rope
```

The target passes when one of these is true:

1. q_norm_rope parity is fixed and the next owner is exposed by rerunning the
   rank-local census.
2. Or q_norm_rope is precisely no-go with a source-parity-backed next port plan.

Do not start CUDA graph, throughput tuning, C128 boundary gates, speculative
acceptance tuning, MoE repair, or indexer FP8 repair in this target unless
q_norm_rope is first classified and closed.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_rank_local_downstream_parity_census/README.md
performance_milestones/target11_mtp_wo_b_projection_reduce_parity/README.md
prompts/TARGET_11.12_dsv4_sm80_mtp_rank_local_downstream_parity_census.md
```

Important TARGET 11.12 result:

```text
rank6 event0/event1:
  q_wqb_output is exact or near-exact
  q_after_q_norm_rope drifts

other ranks:
  later post-all-reduce mismatches are downstream of rank-local upstream drift
```

Do not reopen target-verify metadata, attention/KV, `wo_a`, or `wo_b` unless
the new operator parity framework proves those earlier attributions were wrong.

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/attention/deepseek_v4.py
```

Relevant mini q/RoPE paths:

```text
python/minisgl/models/deepseek_v4.py: around q_wqb_output -> q_after_q_norm_rope
python/minisgl/kernel/deepseek_v4.py: q_norm_rope_fallback
python/minisgl/kernel/deepseek_v4.py: q_kv_norm_rope_cache_fallback
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/elementwise.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/
```

Relevant SGLang q/RoPE behavior:

```text
MQALayer._compute_q_b
fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)
```

Use SGLang source behavior as the preferred contract when mini and SGLang
differ.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id, or
  observed token numerical branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6` as the fix.
- Do not use sequential recompute accepted rows as the final runtime.
- Do not fix MoE or indexer in this target unless q/RoPE is already closed and
  the report explicitly justifies broadening scope.
- Do not rely only on full-generation token mismatches.  The goal is operator
  allclose / first-difference evidence.

## Operator Parity Method

### What "operator drift" means

Drift means:

```text
normal target decode operator output
!=
MTP target-verify operator output
```

under the same visible target prefix/state.  It does not mean either output is
bad relative to natural language quality.  Normal decode can be text-sane while
target verify is still wrong because speculative decoding requires the two
target-model paths to be equivalent.

### Parity record schema

Create or reuse a lightweight record format for each checked operator:

```text
case_id
rank
layer
request_id
verify_event_id
row_depth
position
input_token
operator_name
normal_kernel_or_path
target_verify_kernel_or_path
input_tensor_metadata
output_tensor_metadata
allclose_result
rtol / atol
max_delta / mean_delta
first_differing_index
normal_sample
target_verify_sample
owner_verdict
```

The owner verdict should be one of:

```text
input already drifted
dispatch/path mismatch
same-kernel output drift
reference-oracle mismatch
operator parity pass
insufficient evidence
```

### Micro allclose probes

Yes, micro probes are expected and encouraged.

For an operator whose normal input and target-verify input are equal or within
tolerance, construct a small check that runs only that operator path and compares
the output:

```text
normal decode captured input -> normal operator path -> normal output
same captured input/state -> target-verify operator path -> target output
allclose(normal output, target output)
```

Useful micro levels:

1. Captured-tensor replay:
   - reuse tensors captured during TP8 traces;
   - run normal and target-verify helpers on the same rank.
2. Synthetic one-row replay:
   - construct one row with the same shape, dtype, position, eps, and freqs;
   - avoid loading full model if the helper can run with saved tensors.
3. Same-kernel oracle:
   - force target-verify to call the exact same helper/kernel as normal decode;
   - if this passes, the owner is dispatch/shape path selection.
4. Reference oracle:
   - use a slow torch implementation to define math semantics;
   - use only for diagnosis, not as the promoted final path.

Micro probes should not replace TP8 validation, but they should reduce the
number of expensive full runs.

## Work Plan

### 1. Reproduce The q/RoPE Owner

Use the 11.12 fixed contract:

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
```

Reproduce at least:

```text
bs=1 event0 rank6 q_wqb_output -> q_after_q_norm_rope
bs=2 event0 rank6 q_wqb_output -> q_after_q_norm_rope
```

If q/RoPE no longer reproduces, rerun the rank-local census and follow the new
top owner.

### 2. Add A Minimal Operator Parity Harness

Build the smallest reusable harness that can:

- capture normal decode and target-verify inputs for one operator;
- record dispatch path/kernel name where possible;
- run allclose with dtype-aware tolerances;
- emit the parity record schema above;
- write raw JSON under the milestone directory.

Keep it debug-only.  It can live in test utilities, a milestone script, or
guarded debug hooks; do not make it part of the hot path unless disabled by
default.

### 3. q_norm_rope Source-Parity Table

Before trying many flags, write a source-parity table:

```text
concept
SGLang fused_q_norm_rope behavior
mini normal decode behavior
mini target-verify behavior
same/different/unknown
action
```

Cover:

- q tensor shape and head layout;
- q_lora/q_wqb output dtype and layout;
- q norm eps and reduction dimensions;
- RoPE position value;
- freqs/cos/sin source and dtype;
- q_nope/q_pe split and concatenate order;
- fused q/kv-store path vs standalone q_norm_rope fallback;
- row-batched vs per-row behavior;
- output dtype and rounding.

### 4. Same-Kernel And Reference Oracles

Try these in order:

1. Same-kernel oracle:
   - force target-verify q/RoPE to use the same helper/kernel as normal decode
     for the failing row.
2. Per-row oracle:
   - run q/RoPE one row at a time for target verify if normal decode is one row.
3. Torch/reference oracle:
   - compute q norm + RoPE in a slow explicit implementation and compare both
     paths against it.
4. SGLang-style oracle:
   - if SGLang's fused_q_norm_rope semantics differ from both mini paths, adapt
     the relevant semantics into mini under `sglang_prefill_extend`.

The goal is to decide whether the owner is:

```text
dispatch/path mismatch
row-batched rounding mismatch
freqs/position mismatch
kernel internal numerical mismatch
oracle construction mismatch
```

### 5. Fix q/RoPE Or Produce A Precise No-Go

Preferred fixes:

1. If dispatch differs:
   - make target-verify use the same q/RoPE helper/kernel as normal decode under
     the `sglang_prefill_extend` contract.
2. If row-batched behavior differs:
   - use a row-invariant q/RoPE path for target verify, analogous to the 11.11
     row-invariant `wo_b` fix.
3. If SGLang semantics differ:
   - port/adapt SGLang fused_q_norm_rope semantics for target verify.
4. If kernel internals differ even with the same inputs:
   - document the kernel-level issue and write the next kernel-focused target.

Do not fix by branching on rank, request, batch size, token, or observed values.

### 6. Rerun Focused Validation

After a fix:

```text
q/RoPE operator parity for bs=1 event0 rank6
q/RoPE operator parity for bs=2 event0 rank6
bs=1 targeted trace
bs=2 targeted trace
bs=1/2/4/5/6 exactness matrix
```

If the matrix still fails, rerun enough of the 11.12 census to identify the next
owner.  Expected next candidates are MoE output or indexer FP8.

## Success Criteria

Minimum:

```text
operator parity harness exists and records dispatch/path plus allclose results
q_norm_rope owner is reproduced and classified
SGLang q/RoPE source-parity table is written
accepted commit remains enabled
no batch/rank/request/token special branch is introduced
```

Full:

```text
q_norm_rope parity fixed for the known failing rows
bs=1/2/4/5/6 exact, or next owner is identified by the same operator framework
micro allclose probes are reusable for MoE/indexer follow-up targets
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- q/RoPE cannot be reproduced and the rank-local census must be redone;
- exactness requires a rank/batch/request/token special case;
- same-kernel and reference oracles disagree in a way that points to an invalid
  oracle;
- fixing q/RoPE requires a larger SGLang fused elementwise kernel port;
- q/RoPE is fixed but independent MoE/indexer owners remain, requiring separate
  targets.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_operator_parity_q_norm_rope/README.md
```

Include:

- reproduction of the 11.12 q/RoPE owner;
- operator parity harness design and location;
- q_norm_rope source-parity table against SGLang;
- micro allclose probe results;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- next owner and next focused target if needed, or TARGET 11.3 go/no-go.

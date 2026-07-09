# TARGET 11.21: DSV4 SM80 MTP Target-Verify Row-Depth Producer Parity

## Status

Next after TARGET 11.20.

TARGET 11.20 built the missing baseline-equivalent row-depth oracle for the
current shared failure window and found the first concrete state value mismatch:

```text
bs=4 uid0 event4 committed depth0/depth1
priority-first mismatch on every TP rank: swa.layer1

rank0 depth0 token 582:
  baseline swa.layer1 = b2e0e60546a2f809
  MTP      swa.layer1 = aac7e0b5c10a66b0

rank0 depth1 token 9628:
  baseline swa.layer1 = 969136934b5c24fc
  MTP      swa.layer1 = 108df57ecefcbed3
```

The same event4 depth1 row remains visible in the event8 pre-verify state:

```text
event8 pre-verify swa.layer1 rows
baseline [969136934b5c24fc, 7a497870e99da4a8]
MTP      [108df57ecefcbed3, 7a497870e99da4a8]
```

TARGET 11.20 also proved this is not a logical ownership bug:

```text
positions, SWA locs, full locs, page-table windows, cached/device lengths,
snapshot shapes, and row-depth ownership match baseline.
```

Therefore the remaining owner is producer-side: find where target-verify
row-depth computation first creates a value different from baseline before the
layer1 SWA KV store.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find and fix, or precisely no-go, the first producer-side mismatch that feeds
the `swa.layer1` committed row value for:

```text
bs=4 uid0 event4 depth0 token 582
bs=4 uid0 event4 depth1 correction token 9628
```

Start from baseline-equivalent state and compare baseline greedy vs MTP
target-verify row-depth tensors from layer0 output through the layer1 SWA KV
store.

The target passes when one of these is true:

1. It identifies and fixes the first producer mismatch, and the exactness
   matrix improves without regressing the scoped MoE fix.
2. Or it produces a precise no-go naming the first mismatching producer
   boundary and the SGLang contract needed for the next implementation target.

The target should answer:

```text
Are the target-verify rows that Mini commits for depth0/depth1 computing the
same layer0->layer1 hidden/attention/KV state that baseline greedy computes
after emitting the same visible tokens?
```

If not, name the first boundary that differs.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_row_depth_committed_state_baseline_parity/README.md
performance_milestones/target11_mtp_accepted_commit_lifecycle_state_parity/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
prompts/TARGET_11.20_dsv4_sm80_mtp_row_depth_committed_state_baseline_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Important TARGET 11.20 constraints:

```text
Do not repair attention_wo_b, attention_wo_a, indexer FP8, C128, or page-table
logic before proving the producer-side first mismatch.

Do not reopen lifecycle row ownership: row category/count and logical locs are
already proven equivalent for the anchor.

Do not treat event8 local exactness under Mini state as success: the state is
already baseline-divergent through event4 swa.layer1.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/attention/
python/minisgl/mem_cache/
performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py
performance_milestones/target11_mtp_row_depth_committed_state_baseline_parity/scripts/row_depth_committed_state_oracle.py
```

Likely Mini producer boundaries:

```text
baseline/MTP row input hidden for event4 depth0/depth1
layer0 input hidden
layer0 prenorm output
layer0 attention q/k/v inputs
layer0 attention q/k/v or stored KV row
layer0 attention output
layer0 attention projection / residual output
layer0 post-attention norm / MLP output if it feeds layer1
layer1 input hidden
layer1 SWA/full KV producer input
layer1 SWA/full KV value before store
layer1 SWA/full KV stored row
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Relevant SGLang behavior to inspect and cite:

```text
target verify row-depth hidden-state construction
target verify positions for depth0/depth1 rows
frozen-KV target view and target KV row writes
layer0 attention/KV write behavior under target verify
how accepted/correction rows become baseline-visible state
```

Use SGLang source behavior as the preferred contract. If Mini's direct-write /
snapshot / rollback model differs, prove producer equivalence by baseline hash
comparison.

## Non-Goals

- Do not start graph/perf work.
- Do not directly fix `attention_wo_b`, `attention_wo_a`, or indexer FP8 unless
  the producer-side first mismatch points there.
- Do not directly fix C128 unless `swa.layer1` producer parity is closed and a
  C128 baseline shadow-bank oracle proves C128 is the next source owner.
- Do not undo the TARGET 11.17 MoE row-invariant local fix.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not add parent batch size, uid, event id, row depth, rank, layer, token, or
  prompt-content special branches.
- Do not treat logical loc/seq_len equality as enough; TARGET 11.20 proved
  ownership is equal and value is not.

## Work Plan

### 1. Reproduce The Anchor

Use the TARGET 11.20 authoritative contract:

```text
TP8
/models/DeepSeek-V4-Flash
page_size=256
num_pages=16
draft_len=2
decode_len=8
max_running_req=4
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
accepted commit enabled
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
```

Reproduce:

```text
bs=4 uid0 event4 depth0 token 582
bs=4 uid0 event4 depth1 token 9628
first committed-state value mismatch: swa.layer1
```

If the matrix or anchor moves, follow the new first committed-state mismatch
and explain why the old anchor is stale.

### 2. Add Producer-Side Compact Probes

Extend the row-depth oracle or operator parity harness to collect compact
per-row hashes for baseline greedy and MTP target verify at the same logical
row depths.

For the anchor rows, record:

```text
row category: depth0 accepted / depth1 correction
logical token
logical position
physical loc
rank
layer
boundary name
shape / dtype
value hash
row hash
max_delta if cheap
first differing index if cheap
```

Keep dumps compact. Avoid full tensor dumps unless the hash comparison cannot
locate the boundary.

### 3. Compare Producer Boundary Timeline

Start coarse, then split only the first mismatch.

Initial boundaries:

```text
event4 depth row input hidden
layer0 input hidden
layer0 attention output
layer0 final/residual output
layer1 input hidden
layer1 q/k/v or KV producer input
layer1 SWA KV value before store
layer1 SWA KV stored row
```

If layer0 output is already different, split layer0:

```text
layer0 prenorm
layer0 q_lora / q path
layer0 q_norm_rope
layer0 wkv / KV path
layer0 attention metadata and KV read/write
layer0 merged_attention_output_before_wo
layer0 wo_a / wo_b local and post-reduce
layer0 residual output
layer0 MoE input/output, if layer0 includes MoE on this model path
```

If layer0 output is exact but layer1 SWA KV differs, split layer1 SWA producer:

```text
layer1 input hidden
layer1 attention norm
layer1 q/k/v projection inputs
layer1 q/k/v projected values
layer1 RoPE/position application
layer1 SWA KV pre-store value
layer1 SWA KV stored row
```

### 4. Position / Row-Depth Semantics Check

For every producer boundary, also record:

```text
positions
seq_lens
row depth
active/padded mask
target-verify row order
metadata table indices
attention backend mode
decode vs prefill/extend flag
```

This is important because the stored location is correct but the value could be
computed with the wrong row-depth position or target-verify mode.

### 5. Source-Parity Table Against SGLang

Write a table before any fix:

```text
Concept
SGLang behavior
Mini baseline greedy
Mini MTP target-verify
Verdict / action
```

Cover at least:

```text
target-verify row-depth hidden construction
depth0 accepted row vs depth1 correction row handling
positions/seq_lens for row-depth attention
layer0/1 SWA KV write semantics
attention metadata mode for target-verify rows
snapshot/rollback vs frozen target KV view
```

### 6. Minimal Fix Policy

A minimal fix is allowed only after the first producer mismatch is identified.
Examples:

```text
wrong row-depth hidden input -> fix row packing/order
wrong position/seq_len -> align target-verify position semantics
wrong attention mode -> align decode/prefill/extend metadata with SGLang
wrong KV pre-store value with exact inputs -> fix the local producer kernel/path
stored row differs from pre-store value -> fix store/copy path
```

Validation after a fix:

```text
swa.layer1 depth0/depth1 mismatch is closed
event8 pre-verify swa.layer1 mismatch is closed
full bs=1/2/4/5/6 matrix is rerun
accepted commit stats are reported
TARGET 11.17 MoE pre-reduce sanity is not regressed
```

Do not promote MTP or start graph/perf in this target.

## Validation Gates

Minimum static checks:

```text
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/distributed/impl.py

git diff --check
```

Matrix:

```text
bs=1
bs=2
bs=4
bs=5
bs=6
```

Use the same six prompts from TARGET 11.20 unless the report states why a new
prompt set is required.

Focused gates:

```text
bs=4 uid0 event4 depth0/depth1 producer boundary timeline
bs=4 uid0 event4 layer1 swa.layer1 pre-store and stored-row comparison
bs=4 uid0 event8 pre-verify swa.layer1 comparison
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_row_depth_producer_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
producer-side row-depth oracle description
layer0->layer1 producer boundary timeline
position/row-depth semantics table
SGLang source-parity table
first producer mismatch or precise no-go
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The matrix moves and the bs=4 uid0 event4 `swa.layer1` anchor is no longer
  the first committed-state mismatch.
- The producer-side oracle cannot compare baseline greedy vs MTP target verify
  at matching row depths; document the missing instrumentation.
- The first mismatch is a SGLang target-verify row-depth metadata contract
  larger than one local fix; document the contract and next implementation
  plan.
- A proposed fix passes only by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt content.
- The `swa.layer1` mismatch is closed but the matrix still fails; close with
  the new first owner rather than starting graph/perf.

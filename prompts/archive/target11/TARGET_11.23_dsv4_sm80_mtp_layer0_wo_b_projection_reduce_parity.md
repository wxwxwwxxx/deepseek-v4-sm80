# TARGET 11.23: DSV4 SM80 MTP Layer0 `wo_b` Projection/Reduce Parity

## Status

Next after TARGET 11.22.

TARGET 11.22 implemented a scoped `wo_a` source-parity fix:

```text
Mini target-verify no longer uses _wo_a_bf16_bmm_projection_row_invariant.
Normal decode and target verify both use actual-row cached BF16 BMM for wo_a.
```

The `bs=4 uid0 event4` producer anchor improved at the requested boundary:

```text
layer0.merged_attention_output_before_wo: exact 16/16
layer0.merged_attention_output_after_inverse_rope: exact 16/16
layer0.attention_wo_a_output: exact 16/16
```

The remaining layer0 producer mismatch is now downstream:

```text
layer0.attention_wo_b_local_output_before_reduce: exact 15/16
layer0.attention_wo_b_post_all_reduce_output: exact 0/16
layer0.final_attention_output: exact 0/16
layer1.input: exact 0/16
layer1.kv_after_kv_norm_rope / swa.layer1: exact 0/16
```

This target should determine whether the `wo_b` post-reduce mismatch is:

```text
a true local wo_b projection contract issue;
all-reduce propagation from one drifting rank-depth row;
or an upstream input/layout/dtype mismatch that survived the wo_a fix.
```

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Make layer0 `attention_wo_b_local_output_before_reduce` and
`attention_wo_b_post_all_reduce_output` equivalent between baseline greedy and
MTP target-verify for the `bs=4 uid0 event4` depth0/depth1 anchor, or prove the
precise projection/reduce contract gap that must be fixed next.

The target passes when one of these is true:

1. `wo_b` local and post-reduce outputs become exact for the anchor rows,
   downstream `layer1.swa` improves or closes, and the exactness matrix improves
   without regressing the scoped `wo_a` and MoE fixes.
2. Or the target produces a precise no-go naming the first remaining local
   contribution, all-reduce propagation, dtype/layout, or backend mismatch.

The target should answer:

```text
Given exact wo_a output for the same row-depth inputs, why does wo_b local or
post-all-reduce still differ?
```

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer0_wo_a_projection_contract_parity/README.md
performance_milestones/target11_mtp_target_verify_row_depth_producer_parity/README.md
performance_milestones/target11_mtp_row_depth_committed_state_baseline_parity/README.md
prompts/TARGET_11.22_dsv4_sm80_mtp_layer0_wo_a_projection_contract_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Important TARGET 11.22 constraints:

```text
Do not reopen wo_a unless new evidence proves it regressed.
Do not repair indexer FP8, C128, layer1 SWA store, lifecycle ownership, or graph
perf in this target.
Do not assume post-all-reduce is independently bad until all-rank local
contribution parity is known.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/distributed/impl.py
performance_milestones/target11_mtp_layer0_wo_a_projection_contract_parity/raw/
```

Likely Mini code paths:

```text
wo_b row-parallel projection path
target-verify row-invariant local/reduce flags
_row_invariant_all_reduce, if active
comm.all_reduce label for attn.wo_b
cached BF16 projection weight/layout path
normal decode actual-row projection path
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Relevant SGLang behavior to inspect and cite:

```text
wo_b RowParallelLinear reduce_results behavior
target-verify row tensor shape around wo_b
whether SGLang has any target-verify-only local/reduce branch
dtype/backend used for wo_b on SM80
post-reduce SUM ownership and downstream residual add order
```

Use SGLang source behavior as the preferred contract. If Mini intentionally
differs, prove Mini's contract is bit-exact for baseline-vs-target rows.

## Non-Goals

- Do not start graph/perf work.
- Do not patch indexer FP8, C128, MoE, lifecycle ownership, or layer1 SWA store.
- Do not undo the TARGET 11.22 `wo_a` actual-row projection fix.
- Do not undo the TARGET 11.17 MoE row-invariant local fix.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not add parent batch size, uid, event id, row depth, rank, layer, token, or
  prompt-content special branches.
- Do not patch post-all-reduce by changing communication policy until local
  contributions across all ranks are understood.

## Work Plan

### 1. Reproduce The Post-`wo_a` Anchor

Use the TARGET 11.22 contract:

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
bs=4 uid0 event4 depth0/depth1
layer0.attention_wo_a_output exact 16/16
layer0.attention_wo_b_local_output_before_reduce exact 15/16
layer0.attention_wo_b_post_all_reduce_output exact 0/16
```

If the anchor moves, follow the new first producer mismatch and explain why the
old one is stale.

### 2. All-Rank `wo_b` Local Contribution Census

For every rank and both anchor depths, record:

```text
wo_b input hash
wo_b weight/layout hash or metadata
local matmul output hash
post-all-reduce output hash
shape / dtype / stride / storage offset / contiguity
backend path
reduce label
max_delta / mean_delta
first differing index
```

Build a table:

```text
rank
depth
input exact?
local output exact?
post-reduce output exact?
post-reduce drift explained by which rank-depth local drift?
```

The first question is whether the 13/16 post-reduce owner from TARGET 11.22 is
collective propagation from the one local-drift row, or whether multiple ranks
have hidden local differences after all.

### 3. Direct `wo_b` Projection Oracle

For the same `wo_a` output rows, compare:

```text
Mini baseline normal decode wo_b local projection
Mini target-verify wo_b local projection
candidate actual-row target-verify projection
candidate row-invariant target-verify projection, if active
SGLang-style row-parallel projection reference, if cheap
```

Record the same compact projection metadata as TARGET 11.22:

```text
input hash
weight hash / layout id
input shape / dtype / stride / storage offset / contiguity
backend path
output shape / dtype / stride
row hash
max_delta / mean_delta
first differing index
rank/depth
```

Classify whether any local mismatch is caused by:

```text
target-verify local projection branch
actual-row vs row-invariant accumulation
weight/layout/view difference
BF16/FP32 cast order
row packing / padding / active mask
```

### 4. Post-All-Reduce Contract Check

If local contributions become exact across all ranks but post-reduce still
differs, inspect reduce semantics:

```text
communication backend
label
input dtype
input shape
input stride/contiguity
output dtype
reduce order if observable
target-verify row grouping
row-invariant all-reduce use, if any
```

If only one or a few local contributions differ, do not patch communication;
trace the local projection owner first.

### 5. Source-Parity Table

Write a source-parity table before any fix:

```text
Concept
SGLang behavior
Mini baseline normal decode
Mini MTP target verify
Candidate fix
Verdict
```

Cover at least:

```text
wo_b input tensor shape
row-parallel local projection path
reduce_results / all-reduce behavior
target-verify row-depth packing
dtype / accumulation behavior
weight layout
post-reduce residual add order
```

### 6. Minimal Fix Policy

A minimal fix is allowed if the first owner is clear. Examples:

```text
target-verify local projection path differs -> align with baseline/SGLang
row-invariant all-reduce branch causes mismatch -> align reduce contract
input layout/view differs despite equal values -> canonicalize shape/view
```

Forbidden fixes:

```text
branch on bs=4 / uid0 / event4 / depth / rank / layer0
post-hoc overwrite layer1 SWA rows
change indexer/C128/lifecycle paths without evidence
disable accepted commit
```

### 7. Validate Downstream Closure

After any fix, validate:

```text
layer0.attention_wo_b_local_output_before_reduce exact
layer0.attention_wo_b_post_all_reduce_output exact or explained
layer0.final_attention_output improves/closes
layer1.input improves/closes
layer1.kv_after_kv_norm_rope / swa.layer1 improves/closes
event8 pre-verify swa.layer1 improves/closes
```

Then rerun:

```text
full bs=1/2/4/5/6 exactness matrix
accepted commit stats
TARGET 11.17 MoE pre-reduce sanity or equivalent focused sanity
TARGET 11.22 wo_a exactness sanity
```

If `wo_b` closes but the matrix still fails, close this target with the new
first owner instead of starting graph/perf.

## Validation Gates

Minimum static checks:

```text
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
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

Use the same six prompts from TARGET 11.22 unless the report states why a new
prompt set is required.

Focused gates:

```text
bs=4 uid0 event4 depth0/depth1 wo_b local/reduce oracle
bs=4 uid0 event4 layer0->layer1 downstream timeline
bs=4 uid0 event8 pre-verify swa.layer1 comparison
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer0_wo_b_projection_reduce_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
all-rank wo_b local contribution census
direct wo_b projection oracle
post-all-reduce contract table
source-parity table against SGLang
before/after layer0->layer1 timeline
wo_a sanity result
MoE sanity result
first remaining owner or promotion/no-go verdict
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The post-`wo_a` anchor moves and layer0 `wo_b` is no longer the first
  output-significant producer mismatch.
- The direct `wo_b` oracle cannot compare baseline vs target-verify on the same
  input rows; document missing instrumentation.
- The first mismatch is a broader row-parallel projection/reduce contract that
  needs a split implementation target.
- A proposed fix passes only by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt content.
- `wo_b` becomes exact but `swa.layer1` remains mismatched; close with the new
  first downstream owner.
- The exactness matrix still fails after a safe fix; close with the new first
  owner rather than starting graph/perf.

# TARGET 11.260: DSV4 SM80 MTP layer1 MoE aggregate-before-reduce parity

## Status

Next after TARGET 11.259.

TARGET 11.259 classified the q_wqb-oracle layer10 input line as:

```text
classification: layer10_input_layerN_owner
N: 1
```

Important conclusion from 11.259:

```text
Under MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1, the carried row is exact
through layer1.moe_input.  Layer1 attention, q_wqb/q_norm_rope/KV, wo_a/wo_b,
post-attention residual, router/topk, routed expert output, and shared expert
output are exact.  The first true no-spec baseline mismatch is:

layer1.expert_aggregate_before_reduce
```

Known q_wqb-oracle anchor:

```text
uid: 0
position: 5
full_loc: 3077
swa_loc: 3077
depth: 0
row_to_batch_index: 0
row_to_parent_batch_index: 0
parent_batch_size: 4
layer: 1
operator: expert_aggregate_before_reduce
q_wqb oracle: MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1
```

The reported layer1 MoE operator split is intentionally surprising:

```text
routed_expert_output: exact
shared_expert_output: exact
expert_aggregate_before_reduce: mismatch
```

If the exact same routed and shared tensors are added in the same dtype/order
for the same logical row, the aggregate should also be exact.  Therefore this
target must first prove whether the aggregate mismatch is a real runtime
contract bug or an instrumentation/comparability mismatch.

Do not patch SWA accepted-commit lifecycle, q_wqb gates, logits, sampler,
graph/perf, low-precision paths, or broad MoE execution before this aggregate
contract is classified.

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
debug/mtp/analyze_q_wqb_projection_parity.py
debug/mtp/analyze_layer_input_upstream_parity.py
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
Milestone directories should contain reports, raw outputs, and one-off
artifacts.  If this target needs reusable MoE aggregate analyzers, put them
under `debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/
```

## Contract To Check

For greedy exact MTP, layer1 MoE aggregation must satisfy:

```text
For a logical target row T = (uid, position, full_loc, depth), if:

  routed_expert_output(T) == no-spec routed_expert_output(T)
  shared_expert_output(T) == no-spec shared_expert_output(T)

then:

  expert_aggregate_before_reduce(T)

must equal the no-spec aggregate for T under the same aggregate dtype, cast,
row identity, and addition order.
```

If this does not hold, one of these must be true:

```text
1. the recorded routed/shared tensors are not the actual tensors used by the
   aggregate;
2. the aggregate uses a different row, chunk, parent row, or stale buffer;
3. baseline and target use different dtype/cast/order for the add;
4. instrumentation compares non-equivalent rows;
5. target-verify MoE microbatch has a publication/indexing bug.
```

## Goal

Classify and, if small and generic, fix the layer1
`expert_aggregate_before_reduce` mismatch under the q_wqb target-normal-shape
debug oracle.

The target passes with one of these classifications:

1. `moe_aggregate_instrumentation_owner`: routed/shared are not actually exact
   at the same row used by aggregate, or the analyzer compares mismatched
   records.
2. `moe_aggregate_row_index_owner`: aggregate uses the wrong source row,
   parent row, microbatch chunk row, or output row.
3. `moe_aggregate_dtype_order_owner`: aggregate differs because baseline and
   target use different dtype, cast site, or addition order.
4. `moe_aggregate_publication_owner`: aggregate computes a correct row but
   publishes or records a different row as reduce input.
5. `moe_aggregate_microbatch_contract_owner`: target-verify MoE microbatch
   shape/chunking is incompatible with no-spec MoE aggregate semantics.
6. `moe_aggregate_reduce_input_owner`: aggregate is correct, but the tensor
   passed into final reduce differs from the recorded aggregate.
7. `moe_aggregate_q_wqb_oracle_artifact`: the layer1 aggregate owner appears
   only under q_wqb oracle gates and is not relevant to the default path once
   equivalent tracing is used.
8. `moe_aggregate_fix`: a generic aggregate/reduce-input contract fix lands and
   improves or passes the full exactness matrix.
9. `moe_aggregate_instrumentation_no_go`: current hooks cannot prove which
   tensor/row is used; add the smallest missing probe or write a narrower
   target.

Do not branch on layer1, uid0, position5, full_loc3077, bs6, rank, token,
trace index, or prompt text.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/README.md
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/raw/q_wqb_layer10_input_upstream_analysis.json
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/raw/baseline_layer1_moe_operator_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/raw/q_wqb_layer1_moe_operator_mtp_matrix_1_2_4_5_6.json
prompts/TARGET_11.259_dsv4_sm80_mtp_layer10_input_upstream_parity_after_q_wqb_oracle.md
prompts/TARGET_11.258_dsv4_sm80_mtp_accepted_commit_swa_contract_after_q_wqb_oracle.md
prompts/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1.
Use q_wqb target-normal-shape only as a debug oracle.
Do not promote q_wqb gates.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Do not restore fail-closed accepted commit.
```

Known 11.259 layer1 evidence:

```text
layer1.input: exact
layer1.attention_output: exact
layer1.post_attention_residual: exact
layer1.moe_input: exact
layer1.router_logits: exact
layer1.topk_ids: exact
layer1.topk_weights: exact
layer1.routed_expert_output_raw: exact
layer1.routed_expert_output: exact
layer1.shared_expert_output_raw: exact
layer1.shared_expert_output: exact
layer1.expert_aggregate_before_reduce: first mismatch
layer1.expert_reduce_output: propagated
layer1.moe_output: propagated
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/run_matrix.py
debug/mtp/analyze_layer_input_upstream_parity.py
debug/mtp/analyze_state_parity.py
```

Likely Mini code regions:

```text
_dsv4_target_verify_moe_microbatch_contract
DSV4MoERunner._forward_target_verify_microbatch
DSV4MoERunner._contract_run_once
DSV4MoERunner._contract_run_variant
DSV4MoERunner.forward
routed_for_aggregate / shared / aggregate
expert_aggregate_fp32_add_probe
expert_aggregate_bf16_add_probe
expert_reduce_output reduce input capture
target-verify row_depth / row_to_batch_index / row_to_parent_batch_index metadata
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang to understand target-verify row construction and MoE execution
semantics.  Do not copy broad code before identifying whether Mini's issue is
aggregate indexing, dtype/order, microbatch shape, or instrumentation.

## Non-Goals

- Do not patch SWA accepted-commit copy/restore.
- Do not patch `swa.layer10` store.
- Do not promote q_wqb gates.
- Do not change C4/C128 state lifecycle unless shared target-row metadata is
  proven wrong.
- Do not change routed expert kernels or shared expert kernels if their actual
  aggregate inputs are proven exact.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision work.

## Work Plan

### 1. Reproduce The Aggregate Owner

Use full `1/2/4/5/6` schedule and the q_wqb target-normal-shape debug oracle.

Confirm against the true no-spec baseline, not the same-run normal-oracle view:

```text
layer1.moe_input exact;
layer1.routed_expert_output exact;
layer1.shared_expert_output exact;
layer1.expert_aggregate_before_reduce mismatch.
```

Also report the default-path first owner as control.  Do not make the q_wqb
oracle path the default interpretation for MTP unless evidence requires it.

### 2. Capture Actual Aggregate Inputs

Add or use probes that capture the exact tensors passed to the aggregate add:

```text
actual routed_for_aggregate tensor;
actual shared tensor;
actual aggregate output tensor;
actual tensor passed to expert_reduce_output;
dtype/shape/stride/storage_offset/data_ptr where useful;
row index within flat, microbatch chunk, and parent batch;
row_depth, row_to_batch_index, row_to_parent_batch_index, active/padded flags.
```

Compare the recorded `routed_expert_output` and `shared_expert_output` against
the actual aggregate inputs.  If they differ, classify
`moe_aggregate_instrumentation_owner` or `moe_aggregate_publication_owner`
before patching math.

### 3. Recompute Aggregate Oracles

For the carried row and all active rows in the same microbatch, compute and
compare:

```text
baseline actual aggregate;
target actual aggregate;
target recompute routed + shared in fp32;
target recompute routed.to(hidden_dtype) + shared.to(hidden_dtype), then fp32;
target recompute shared + routed;
target per-row aggregate;
target parent-batch-shaped aggregate if metadata is available;
baseline per-row aggregate;
baseline parent-batch-shaped aggregate.
```

Record which oracle matches the no-spec baseline and which matches the current
target runtime.  This should decide dtype/order versus row-shape/indexing.

### 4. Verify Row And Chunk Mapping

For the target-verify MoE microbatch contract, report:

```text
contract.rows;
contract.chunk_rows;
contract.row_order;
contract.verify_width;
contract.active_row_mask;
contract.padded_row_mask;
contract.source_rows per chunk;
flat row -> parent batch row;
flat row -> request row;
flat row -> depth;
flat row -> output row after contract_run_variant;
```

Check whether `_contract_run_variant` concatenates chunk outputs in the same
row order that `hidden_states.view_as` and downstream layer inputs expect.

### 5. Split Reduce Input Publication

Even if `expert_aggregate_before_reduce` is recorded as mismatched, verify
whether the final reduce input is the same tensor:

```text
aggregate recorded output;
clone_operator_row0_input("expert_reduce_output");
pre_reduce tensor passed into maybe_reduce_final;
post-reduce tensor;
moe_output final cast input/output.
```

If aggregate is correct but reduce input is wrong, classify
`moe_aggregate_reduce_input_owner`.

### 6. Minimal Generic Fix Policy

If the violated contract is clear and the fix is small, it may be implemented.
Prefer fixes that:

```text
make aggregate row publication metadata-driven;
align target-verify microbatch row order with no-spec target row order;
make dtype/cast/order explicit and shared between baseline and target-verify;
reuse SGLang-compatible target-verify MoE semantics where clear;
preserve the 11.246 MoE microbatch guard.
```

Avoid fixes that:

```text
special-case layer1, uid0, bs6, or full_loc3077;
hide mismatches by skipping analyzer rows;
disable target verify or accepted commit;
promote q_wqb gates as an MoE workaround;
rewrite MoE broadly without proving the aggregate owner.
```

### 7. Validation

If runtime code changes, validate:

```text
full 1/2/4/5/6 exactness matrix;
text sanity smoke;
TARGET 11.246 MoE microbatch guard;
TARGET 11.249/11.250 C128 guards;
TARGET 11.251/11.252 analyzer validity guards;
TARGET 11.253 SWA commit/copy guard;
TARGET 11.256 q_wqb owner guard;
TARGET 11.257 q_wqb oracle evidence remains explainable;
TARGET 11.258 SWA accepted-commit lifecycle remains faithful;
TARGET 11.259 layer-input bisection remains explainable.
```

Also run static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/analyze_q_wqb_projection_parity.py \
  debug/mtp/analyze_layer_input_upstream_parity.py \
  debug/mtp/run_matrix.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

## Stop Conditions

Stop this target when one of these is true:

1. The layer1 `expert_aggregate_before_reduce` mismatch is classified into
   instrumentation, row index, dtype/order, publication, microbatch contract,
   reduce input, or q_wqb-oracle artifact with evidence.
2. A generic aggregate/reduce-input fix lands and the exactness matrix improves
   or passes.
3. The aggregate owner is disproven and the next true owner is named.
4. Instrumentation is insufficient and the missing tensor/row/checkpoint is
   specified exactly.

Do not continue into graph/perf or broad non-MTP optimization after the
aggregate contract segment is classified.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/README.md
```

Include:

- final classification;
- default-path versus q_wqb-oracle control summary;
- exact aggregate contract statement;
- actual aggregate input table;
- recomputed aggregate oracle table;
- row/chunk mapping table;
- reduce input publication table;
- SGLang source-parity notes;
- any code changes and gates;
- before/after exactness matrix if a fix lands;
- commands and tests run;
- next target recommendation if exactness still fails.

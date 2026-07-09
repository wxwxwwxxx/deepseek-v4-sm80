# TARGET 11.261: DSV4 SM80 MTP rank2 layer1 shared expert parity after q_wqb oracle

## Status

Next after TARGET 11.260.

TARGET 11.260 reclassified the previous layer1 MoE aggregate owner as an
instrumentation owner:

```text
classification: moe_aggregate_instrumentation_owner
```

The old target `expert_aggregate_before_reduce` record was captured after the
in-place final all-reduce had already mutated the tensor.  After preserving the
actual pre-reduce snapshot, rank0 aggregate and reduce input match the true
no-spec baseline.  The all-rank table now points to the first true owner:

```text
rank2 layer1.shared_expert_output
```

Observed under `MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1` and
`MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1`:

```text
rank2 routed_expert_output: exact
rank2 shared_expert_output: mismatch
rank2 expert_aggregate_before_reduce: propagated
rank2 expert_reduce_output input: propagated
post-reduce output on all ranks: propagated from rank2
```

Do not patch aggregate, final all-reduce, SWA accepted-commit lifecycle, q_wqb
gates, logits, sampler, graph/perf, communication policy, or low-precision paths
before the rank2 shared-expert boundary is classified.

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
debug/mtp/analyze_moe_aggregate_parity.py
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
If this target needs a reusable shared-expert analyzer, put it under
`debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle/
```

## Contract To Check

For greedy exact MTP, the rank-local layer1 shared expert must satisfy:

```text
For a logical target row T = (uid, position, full_loc, depth), on each TP rank R:

  shared_expert_input_R(T) == no-spec shared_expert_input_R(T)
  shared expert weights/cache/backend for R are equivalent
  row mapping and source row used by the shared expert are equivalent

then:

  shared_expert_output_raw_R(T)
  shared_expert_output_R(T)

must equal the no-spec rank-local shared-expert outputs for T, before the final
MoE aggregate and before any propagated all-reduce output is considered.
```

The shared expert is part of the rank-local pre-reduce MoE contribution.  A
post-reduce mismatch on every rank is not evidence against the communication
stack if one rank has already contributed a wrong shared-expert row.

## Goal

Classify and, if small and generic, fix the rank2
`layer1.shared_expert_output` mismatch under the q_wqb target-normal-shape debug
oracle.

The target passes with one of these classifications:

1. `shared_expert_input_owner`: rank2 shared expert sees a different input row
   than true no-spec baseline.
2. `shared_expert_row_mapping_owner`: the input value is equivalent in a trace,
   but the runtime shared expert consumes the wrong row, source row, chunk row,
   parent row, active mask, or padded row.
3. `shared_expert_gate_up_owner`: mismatch first appears at
   `shared_experts.gate_up_proj`.
4. `shared_expert_activation_owner`: gate/up output is exact, but
   `silu_and_mul_clamp_fallback` or the hidden cast into down projection first
   differs.
5. `shared_expert_down_proj_owner`: mismatch first appears at
   `shared_experts.down_proj`.
6. `shared_expert_finalize_cast_owner`: raw shared output is exact, but the
   fp32/finalized shared output differs.
7. `shared_expert_weight_cache_owner`: shared expert weights, BF16 caches, or
   dense FP8 Marlin cache are not equivalent for rank2/layer1.
8. `shared_expert_backend_row_shape_owner`: the selected backend is row-shape
   sensitive, and target-verify microbatch shape differs from no-spec shape.
9. `shared_expert_backend_data_sensitivity_owner`: the same backend is
   deterministic for repeated identical inputs, but specific value ranges from
   the failing row expose unacceptable drift versus reference.
10. `shared_expert_backend_nondeterminism_owner`: repeated identical inputs,
    shape, weights, and backend produce different outputs.
11. `shared_expert_recording_owner`: recorded shared tensors are not the actual
   tensors consumed by aggregate.
12. `shared_expert_sglang_contract_owner`: Mini's target-verify shared-expert
    contract differs materially from SGLang's DSV4/MTP behavior.
13. `shared_expert_fix`: a generic shared-expert contract fix lands and improves
    or passes the full exactness matrix.
14. `shared_expert_no_go`: current hooks cannot prove the owner; write the
    missing probe or a narrower next target.

Do not branch on layer1, rank2, uid0, position5, full_loc3077, bs6, token, trace
index, or prompt text.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/README.md
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/raw/q_wqb_layer1_moe_aggregate_analysis_after_pre_reduce_snapshot.json
performance_milestones/target11_mtp_layer1_moe_aggregate_before_reduce_parity/raw/q_wqb_layer1_moe_operator_mtp_matrix_1_2_4_5_6_after_pre_reduce_snapshot.json
performance_milestones/target11_mtp_layer10_input_upstream_parity_after_q_wqb_oracle/raw/baseline_layer1_moe_operator_matrix_1_2_4_5_6.json
prompts/TARGET_11.260_dsv4_sm80_mtp_layer1_moe_aggregate_before_reduce_parity.md
prompts/TARGET_11.259_dsv4_sm80_mtp_layer10_input_upstream_parity_after_q_wqb_oracle.md
prompts/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Known 11.260 evidence:

```text
rank0 routed/shared/aggregate/reduce-input: exact
rank1 routed/shared/aggregate/reduce-input: exact
rank2 routed: exact
rank2 shared: mismatch
rank2 aggregate/reduce-input: propagated
all ranks post-reduce: propagated
```

Known q_wqb-oracle anchor:

```text
uid: 0
input token: 11111
position: 5
full_loc: 3077
swa_loc: 3077
depth: 0
layer: 1
rank: 2 is the current owner
target flat rows: 3
live target batch size: 1
parent batch size: 4
verify width: 3
chunk rows: 1
row order: request_major
source rows per chunk: [0], [1], [2]
active row mask: [true, true, true]
padded row mask: [false, false, false]
row_to_batch_index: [0, 0, 0]
row_to_parent_batch_index: [0, 0, 0]
q_wqb oracle: MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1
MoE microbatch runtime: MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Keep q_wqb target-normal-shape as a debug oracle only.
Do not promote q_wqb gates.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Preserve TARGET 11.260 pre-reduce snapshot instrumentation.
Do not restore fail-closed accepted commit.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/dense_fp8_marlin.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/run_matrix.py
debug/mtp/analyze_moe_aggregate_parity.py
```

Likely Mini code regions:

```text
DSV4SharedExperts.forward
DSV4SharedExperts.prepare_bf16_weight_cache
DSV4SharedExperts.prepare_down_marlin_weight_cache
DSV4MoERunner.apply_shared_raw
DSV4MoERunner.finalize_shared
DSV4MoERunner._contract_run_once
DSV4MoERunner._contract_run_variant
DSV4MoERunner._forward_target_verify_microbatch
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION
MINISGL_DSV4_SM80_SHARED_FP8_GEMM
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang to understand whether target-verify shared expert is executed on the
same gathered/local row surface as the normal target model, and whether any
local/shared-expert optimization changes the expected row shape.  Prefer source
parity over inventing a Mini-only contract.

## Non-Goals

- Do not patch `expert_aggregate_before_reduce`.
- Do not patch `expert_reduce_output` or final MoE all-reduce.
- Do not patch SWA accepted-commit copy/restore.
- Do not promote q_wqb target-normal-shape gates.
- Do not change C4/C128 state lifecycle unless shared-expert input metadata is
  proven wrong.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision work.

## Work Plan

### 1. Reproduce The Rank2 Shared Owner

Use the full `1/2/4/5/6` schedule and the q_wqb target-normal-shape debug
oracle.

Confirm against the true no-spec baseline, not the same-run normal-oracle view:

```text
rank2 layer1.shared_expert_input: expected exact or classify input owner
rank2 layer1.shared_expert_output_raw: compare
rank2 layer1.shared_expert_output: first known mismatch
rank2 layer1.expert_aggregate_before_reduce: propagated
rank2 layer1.expert_reduce_output input: propagated
```

Also report the default-path first owner as control.  Do not make the q_wqb
oracle path the default interpretation for MTP unless evidence requires it.

### 2. Prove The Actual Shared Input Row

Capture the exact tensor row consumed by `DSV4SharedExperts.forward` on rank2:

```text
shared_expert_input checksum/value sample;
shape, stride, storage_offset, dtype, device, data_ptr;
logical row id: uid, position, full_loc, depth;
flat row index, source row, chunk row, parent batch row;
active/padded flags and row_to_batch/row_to_parent_batch mapping;
whether the input is contiguous and whether a copy was made before gate_up.
```

Compare true no-spec baseline and target for the same logical row.  If the input
is already different, stop with `shared_expert_input_owner` and do not inspect
downstream shared expert math.

### 3. Split Shared Expert Internals

For rank2/layer1, add or reuse probes for:

```text
gate_up input;
gate_up output before chunk;
gate chunk;
up chunk;
silu_and_mul_clamp_fallback output;
hidden_for_down after dtype cast;
down_proj output before any finalization;
shared_expert_output_raw;
shared_expert_output after fp32 finalize.
```

Record dispatch/backend metadata:

```text
use_bf16_weight_cache;
dense_fp8_marlin_projection_enabled;
MINISGL_DSV4_SM80_SHARED_FP8_GEMM;
gate_up cache owner/name;
down cache owner/name;
down Marlin cache owner/name;
weight/cache dtype and shape;
reduce flag used by shared expert;
row_invariant_local flag.
```

Classify the first divergent internal boundary.  Avoid broad kernel changes
until this split names gate_up, activation, down_proj, finalize, input, cache,
or recording as the owner.

### 4. Test Row-Shape Oracles

The target-verify MoE microbatch currently has three flat rows and executes
chunks of one row.  Test whether shared expert is row-shape sensitive:

```text
baseline normal target shape;
target current chunk_rows=1;
target full flat rows=3;
target parent-like rows=4 if the parent row surface is reconstructable;
target active-only rows;
target row-by-row reference;
target repeated-row probes if useful to isolate backend row sensitivity.
```

For each oracle, state whether it matches:

```text
true no-spec baseline;
current target runtime;
same-run target normal-shape oracle;
SGLang-equivalent target-verify contract, if known.
```

If only a shape-compatible oracle is exact, classify
`shared_expert_backend_row_shape_owner` or `shared_expert_sglang_contract_owner`
before implementing a runtime fix.

### 5. Run A Targeted Stability/Data-Sensitivity Microbench

Do not start with broad random fuzzing.  First extract the real rank2/layer1
failing shared-expert input row and backend/cache metadata, then replay only the
smallest useful variants.

Required probes:

```text
same input, same shape, same backend, repeated N times;
same logical row as 1 row, 3 target rows, parent-like 4 rows if reconstructable;
same logical row with padded/inactive neighbor rows;
same logical row through reference/torch path if available;
gate_up-only, activation-only, down_proj-only, and full shared expert.
```

Report:

```text
max_abs / max_rel / checksum across repeats;
whether output changes across repeated identical runs;
whether output changes only when row shape or neighbor rows change;
whether the real failing row differs from nearby synthetic rows;
whether simple value sweeps reproduce the drift.
```

Optional value sweeps should be small and anchored to the captured failing row:

```text
zeros;
small scale;
original row;
2x / 4x scale;
sign-flipped row;
clamp-boundary-biased row if gate/up values show clamp saturation.
```

Use this only to classify backend behavior.  Do not promote random-test-derived
patches unless they also fix the true no-spec baseline parity.

### 6. Verify Weight/Cache/Backend Equivalence

Do a small rank2/layer1 weight/cache ledger:

```text
gate_up original FP8/e8m0 weight metadata;
gate_up BF16 cached weight metadata if enabled;
down original FP8/e8m0 weight metadata;
down BF16 cached weight metadata if enabled;
down dense FP8 Marlin cache metadata if enabled;
whether original weights were released or retained;
whether baseline and target paths read the same cache object/owner.
```

If necessary, run diagnostic toggles as oracles only:

```text
disable shared expert BF16 weight cache if possible;
disable dense FP8 Marlin projection if possible;
disable shared FP8 GEMM if possible;
force row-invariant/local or normal-shape replay only as a debug comparison.
```

Do not promote or remove any backend based only on a single oracle.  The target
must explain whether a backend is truly non-equivalent or only being called with
the wrong target-verify shape.

### 7. Compare Against SGLang

Review SGLang's DeepSeek V4 shared expert and MTP target-verify path:

```text
whether shared expert is computed before/after gather/scatter;
whether shared expert sees local rows or a gathered global row surface;
whether MTP target verify reuses normal target row shape;
whether shared expert fusion/local optimization is enabled or disabled;
where output is added back relative to MoE reduce/scatter.
```

If SGLang has an explicit target-verify shared-expert contract that Mini lacks,
prefer adapting that contract over adding a Mini-only special case.

### 8. Minimal Generic Fix Policy

If the violated contract is clear and the fix is small, it may be implemented.
Prefer fixes that:

```text
make target-verify shared expert consume the same logical/shape surface as no-spec baseline;
align microbatch row order and chunking with SGLang-compatible semantics;
make shared expert dtype/cast/finalization explicit and shared between paths;
preserve the 11.246 MoE microbatch guard;
preserve 11.260 pre-reduce snapshot correctness instrumentation.
```

Avoid fixes that:

```text
special-case rank2, layer1, uid0, bs6, full_loc3077, or token 11111;
hide mismatches by skipping analyzer rows;
disable shared expert, target verify, or accepted commit;
promote q_wqb gates as a shared-expert workaround;
rewrite all MoE execution before proving the shared-expert owner.
```

### 9. Validation

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
TARGET 11.260 aggregate instrumentation owner remains explainable.
```

Also run static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/analyze_q_wqb_projection_parity.py \
  debug/mtp/analyze_layer_input_upstream_parity.py \
  debug/mtp/analyze_moe_aggregate_parity.py \
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

1. The rank2 layer1 `shared_expert_output` mismatch is classified into input,
   row mapping, gate_up, activation, down_proj, finalize, weight/cache, backend
   row-shape, backend data sensitivity, backend nondeterminism, recording, or
   SGLang contract owner with evidence.
2. A generic shared-expert contract fix lands and the exactness matrix improves
   or passes.
3. The rank2 shared-expert owner is disproven and the next true owner is named.
4. Instrumentation is insufficient and the missing tensor/row/checkpoint is
   specified exactly.

Do not continue into graph/perf or broad non-MTP optimization after the
rank2 shared-expert segment is classified.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_rank2_layer1_shared_expert_parity_after_q_wqb_oracle/README.md
```

Include:

- final classification;
- default-path versus q_wqb-oracle control summary;
- exact shared-expert contract statement;
- rank2 shared input equivalence table;
- shared expert internal boundary table;
- row-shape oracle table;
- targeted stability/data-sensitivity microbench table;
- weight/cache/backend ledger;
- SGLang source-parity notes;
- any code changes and gates;
- before/after exactness matrix if a fix lands;
- commands and tests run;
- next target recommendation if exactness still fails.

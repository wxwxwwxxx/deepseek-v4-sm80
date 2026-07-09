# TARGET 11.255: DSV4 SM80 MTP Layer0 Output Sub-Boundary Parity

## Status

Next after TARGET 11.254.

TARGET 11.254 classified the first producer-side owner as:

```text
classification: producer_layer0_output_owner
anchor: uid0 / position5 / full_loc3077
first non-equivalent boundary: layer0.post_moe_residual
```

Important conclusion from 11.254:

```text
metadata is aligned;
embedding/model input is exact;
layer0 input is exact;
layer0.post_moe_residual is the first observed mismatch;
layer1 differences only propagate the already-diverged layer0 output.
```

The next target must split layer0 internally.  Do not reopen SWA accepted
commit, layer1 store, C4 indexer state, C128 raw-loc mapping, logits, sampler,
graph, or communication paths unless the layer0 split disproves previous
findings.

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
artifacts.  If this target needs a reusable helper, put it under `debug/mtp/`
and write outputs under:

```text
performance_milestones/target11_mtp_layer0_output_subboundary_parity/
```

## Goal

Find the first layer0 sub-boundary where MTP target-verify correction row
`uid0/position5/full_loc3077` diverges from no-spec normal decode, given that
the layer0 input is exact and `layer0.post_moe_residual` is not.

Split at least these layer0 boundaries:

```text
layer0.input;
layer0 attention input / normed input if present;
layer0 q / q_lora / q_norm_rope;
layer0 KV projection / KV norm/RoPE if present;
layer0 consumed attention state and page/SWA/C4/C128 metadata;
layer0 attention raw output;
layer0 wo_a / wo_b projection and any all-reduce;
layer0 post-attention residual;
layer0 MoE input;
layer0 router/topk;
layer0 routed expert output;
layer0 shared expert output;
layer0 MoE pre/post reduce;
layer0 post-MoE residual.
```

The target passes with one of these classifications:

1. `layer0_attention_metadata_owner`: layer0 input is exact, but consumed
   attention metadata/state differs.
2. `layer0_attention_qkv_owner`: attention metadata is aligned, but Q/KV
   projection, norm, or RoPE first differs.
3. `layer0_attention_output_owner`: Q/KV and metadata are aligned, but attention
   output first differs.
4. `layer0_attention_wo_owner`: attention raw output is aligned, but `wo_a`,
   `wo_b`, or attention output reduce first differs.
5. `layer0_post_attention_residual_owner`: attention projection output is
   aligned, but residual add/norm materialization differs.
6. `layer0_moe_input_owner`: post-attention row differs before MoE, making MoE a
   downstream propagation path.
7. `layer0_moe_router_owner`: MoE input is aligned, but router/topk differs.
8. `layer0_moe_expert_owner`: router/topk is aligned, but routed/shared expert
   output differs.
9. `layer0_moe_reduce_owner`: pre-reduce expert aggregate is aligned, but
   reduce/post-reduce differs.
10. `layer0_post_moe_residual_owner`: MoE output is aligned, but final residual
    add/materialization differs.
11. `layer0_metadata_owner`: values are aligned when compared directly, but
    token, position, out_loc, full_loc, SWA loc, depth, or row metadata is wrong.
12. `layer0_instrumentation_no_go`: current hooks cannot split layer0 enough;
    add the smallest missing instrumentation or write a narrower target.

If a minimal source-aligned fix is clear, it may be attempted.  Otherwise stop
with the precise owner and next repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/README.md
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/README.md
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/README.md
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/README.md
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
prompts/TARGET_11.254_dsv4_sm80_mtp_layer1_swa_producer_boundary_parity.md
prompts/TARGET_11.253_dsv4_sm80_mtp_swa_layer1_commit_row_value_parity.md
prompts/TARGET_11.252_dsv4_sm80_mtp_c4_indexer_state_validity_consume.md
prompts/TARGET_11.251_dsv4_sm80_mtp_online_c128_parity_planner_next_owner.md
prompts/TARGET_11.250_dsv4_sm80_mtp_online_c128_read_surface_port.md
prompts/TARGET_11.249_dsv4_sm80_mtp_online_c128_main_state_contract_port.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Key artifacts:

```text
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/boundary_summary.json
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/baseline_matrix_1_2_4_5_6_producer_trace.json
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/mtp_matrix_1_2_4_5_6_producer_trace.json
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Preserve TARGET 11.249/11.250 C128 main-state storage/read-surface behavior
unless source parity proves it is wrong.
Preserve TARGET 11.251/11.252 analyzer validity rules unless this target proves
they are wrong.
Do not restore fail-closed accepted commit as a way to pass exactness.
Do not reopen MoE microbatch work unless focused guards prove it regressed.
Do not patch logits/sampler.
Do not start CUDA graph/perf, PyNCCL, communication-policy work, or low
precision research.
Do not branch on uid, position, layer, loc, bs, request id, token, rank, or
prompt text.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

Likely paths to audit:

```text
DeepSeekV4 layer forward;
target-verify batch construction;
target-verify row/depth metadata;
attention backend metadata and consumed KV/SWA/C4/C128 state;
wo_a / wo_b projection and reduce;
MoE router/topk, expert execution, shared expert, aggregate, reduce;
residual add/norm/materialization;
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH runtime path.
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

## Non-Goals

- Do not change SWA accepted commit/copy/restore unless this target disproves
  TARGET 11.253.
- Do not change layer1 producer/store unless this target disproves TARGET
  11.254.
- Do not change C4 indexer state runtime; TARGET 11.252 classified it as
  uninitialized/unconsumed.
- Do not change C128 main-state storage/read surface unless source parity proves
  it is wrong.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf or low-precision work.

## Work Plan

### 1. Reproduce The Layer0 Owner

Use the full `1/2/4/5/6` schedule artifacts, not isolated bs6, because 11.254
showed isolated bs6 uses allocator loc `5` rather than the full-schedule
`3077`.

Confirm:

```text
metadata aligned for uid0/position5/full_loc3077;
layer0.input exact;
layer0.post_moe_residual mismatch;
accepted commit active and not fail-closed;
C128 raw-loc and C4 uninitialized-state owners remain filtered.
```

### 2. Add Focused Layer0 Sub-Boundary Trace

Add or use debug trace hooks for the anchor row in both no-spec baseline and MTP
target verify:

```text
uid/request identity;
token id;
position;
full_loc;
swa_loc;
target-verify depth/category;
layer id;
sub-boundary name;
shape/dtype;
checksum / abs_sum / sample;
rank id;
rank-local or post-reduce/global status.
```

Required sub-boundary labels, or nearest local equivalents:

```text
layer0.input;
layer0.attention_input;
layer0.q_state;
layer0.kv_state_or_wkv;
layer0.q_norm_rope;
layer0.kv_norm_rope;
layer0.attention_metadata;
layer0.attention_output_raw;
layer0.attention_wo_a_output;
layer0.attention_wo_b_output_or_post_reduce;
layer0.post_attention_residual;
layer0.moe_input;
layer0.router_logits;
layer0.topk_ids_and_weights;
layer0.routed_expert_output;
layer0.shared_expert_output;
layer0.moe_pre_reduce;
layer0.moe_post_reduce;
layer0.post_moe_residual.
```

### 3. Metadata And Consumed-State Parity

Before classifying numeric owners, verify layer0 consumed state/metadata:

```text
token / position / out_loc / full_loc / swa_loc;
attention page table window;
SWA consumed locs and values;
C4/C128 consumed state metadata, using TARGET 11.251/11.252 validity rules;
target-verify frozen-KV read-only behavior;
row/depth/category metadata;
rank-local row mapping.
```

If metadata differs, classify `layer0_metadata_owner`.

### 4. Attention Versus MoE Split

First decide whether the first mismatch occurs before MoE:

```text
If attention output or post-attention residual differs, stop in the attention
branch.

If post-attention residual and MoE input are exact but MoE output differs, stop
in the MoE branch.

If MoE output is exact but post-MoE residual differs, stop at residual
materialization.
```

Do not deep-dive both attention and MoE in the same run unless the split is
ambiguous.  The target should produce a smaller next repair target once the
branch is known.

### 5. MoE-Specific Guard

Because earlier TARGET 11.246 introduced normal-shape-compatible target-verify
MoE microbatching, keep:

```text
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
```

If the first mismatch appears in MoE, explicitly compare:

```text
current runtime microbatch path;
the old MoE exactness guard from TARGET 11.246;
whether this anchor uses a row shape or schedule not covered by the 11.246
focused guard.
```

Do not disable microbatching as a fix unless a source-parity proof shows it is
wrong for this anchor and the replacement path passes focused guards.

### 6. Minimal Fix Policy

Allowed fixes:

```text
target-verify metadata;
row/depth flatten/unflatten order;
attention consumed-state metadata;
normal-shape microbatch scheduling for a proven MoE boundary;
projection/reduce source or destination if proven wrong;
trace/analyzer boundary semantics if needed.
```

Forbidden fixes:

```text
special-casing uid0/position5/layer0/full_loc3077/bs6;
changing logits/sampler;
disabling accepted commit or target verify;
restoring fail-closed behavior;
loosening exactness;
undoing C128/C4 analyzer fixes without evidence.
```

### 7. Validation

If runtime code changes, run:

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
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
batch sizes: 1 2 4 5 6
```

Use `debug/mtp/run_matrix.py` unless a target-specific runner is required.

Required checks if runtime changes:

```text
bs1/2/4/5/6 greedy exactness matrix;
MTP text sanity;
non-MTP baseline text sanity;
11.246 MoE microbatch focused guard if target verify row execution changes;
11.247 accepted-commit state/KV guard;
11.253 SWA lifecycle anchor;
11.254 producer boundary anchor;
C128 online lifecycle remains active and not fail-closed;
C4 indexer-state skip remains valid in analyzer.
```

Static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/run_matrix.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py

git diff --check
```

If only trace instrumentation changes and runtime behavior is default-off or
unchanged, rerun only the focused traced matrix and clearly state exactness is
unchanged.

## Stop Conditions

Stop and write the milestone report when one of these happens:

1. The first layer0 sub-boundary is classified with enough evidence for a repair
   target.
2. A minimal source-aligned fix makes bs1/2/4/5/6 exact.  Run all required
   guards and recommend the next MTP promotion target.
3. Metadata or consumed-state parity differs.  Stop with
   `layer0_metadata_owner` and write the smallest metadata repair target.
4. Current hooks cannot split layer0 enough.  Add small instrumentation if
   possible; otherwise stop with a precise instrumentation target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer0_output_subboundary_parity/README.md
```

The report must include:

- final classification;
- metadata and consumed-state parity table for `uid0/position5/full_loc3077`;
- layer0 sub-boundary comparison table;
- attention-vs-MoE branch decision;
- source-parity notes against SGLang;
- any implementation changes;
- exactness matrix and smoke results if runtime changed;
- tests/static checks;
- remaining risks and next target recommendation.

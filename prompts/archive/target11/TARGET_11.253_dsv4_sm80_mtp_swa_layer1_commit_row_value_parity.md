# TARGET 11.253: DSV4 SM80 MTP SWA Layer1 Commit Row Value Parity

## Status

Next after TARGET 11.252.

TARGET 11.252 proved that `c4_indexer_state.layer10` is not a valid runtime
correctness owner:

```text
classification: c4_indexer_state_uninitialized_skip
reason: analyzer-visible but not trace-visible written or consumed;
        live C4 indexer cache/FP8 surfaces are exact.
```

After the C128-aware planner and C4 uninitialized-state skip, the first valid
consumed owner is:

```text
component: swa.layer1
owner: commit_row_value_owner
baseline trace: 100, baseline_after_normal_decode
MTP trace: 117, mtp_after_accepted_commit
uid: 0
position: 5
full_loc: 3077
mapping: aligned
baseline sha: 6e69ea3207e2a07e
baseline abs_sum: 299.7920684814453
MTP sha: fa3cb29bdfd68314
MTP abs_sum: 299.9810028076172
baseline sample: [-0.267578125, 0.1337890625, -0.546875, 0.1650390625]
MTP sample:      [-0.267578125, 0.1318359375, -0.5546875, 0.154296875]
```

The row mapping is aligned and the values are finite.  This is now a credible
runtime lifecycle/value owner, unlike the C128 raw-loc owner and C4
uninitialized-state owner.

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
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/
```

## Goal

Determine why the accepted-commit path leaves `swa.layer1` at
`uid0/position5/full_loc3077` with a value different from no-spec baseline.

Classify the owner as one of:

1. `swa_target_verify_producer_value_owner`: the target-verify row value is
   already different before accepted commit.
2. `swa_commit_copy_source_owner`: accepted commit copies from the wrong source
   row/depth/bank.
3. `swa_commit_copy_destination_owner`: accepted commit writes to the wrong
   destination row, even though the source is correct.
4. `swa_commit_copy_kernel_value_owner`: source and destination mapping are
   correct, but the copy/move writes different values.
5. `swa_snapshot_restore_owner`: the row is correct after copy and later
   snapshot/restore, rejected-tail cleanup, or rollback changes it.
6. `swa_baseline_alignment_owner`: the baseline comparison row is not the
   no-spec-equivalent row under the current target-verify schedule.
7. `swa_instrumentation_no_go`: current traces cannot split producer, copy, and
   restore; add the smallest missing instrumentation or write the next
   instrumentation target.

If a minimal fix is clear and source-aligned, it may be attempted.  Otherwise
stop with the precise owner and next repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/README.md
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/raw/
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/README.md
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
prompts/TARGET_11.252_dsv4_sm80_mtp_c4_indexer_state_validity_consume.md
prompts/TARGET_11.251_dsv4_sm80_mtp_online_c128_parity_planner_next_owner.md
prompts/TARGET_11.250_dsv4_sm80_mtp_online_c128_read_surface_port.md
prompts/TARGET_11.249_dsv4_sm80_mtp_online_c128_main_state_contract_port.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Key artifacts:

```text
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/raw/analysis_bs6_c4_validity.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/mtp_matrix_1_2_4_5_6_prefill_bank0.json
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
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not patch logits/sampler.
Do not start CUDA graph/perf, PyNCCL, communication-policy work, or low
precision research.
Do not branch on uid, position, layer, loc, bs, request id, token, rank, or
prompt text.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

Likely paths to audit:

```text
accepted commit / target_commit_kv copy path
_verify_mtp_spec_drafts_flattened
_snapshot_mtp_kv_rows
_record_mtp_state_parity / state trace hooks
SWA loc mapping and component ownership
target-verify temporary KV/SWA writes
target-verify row/depth metadata
rejected-tail isolation and cleanup
snapshot/restore around target verify
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

## Non-Goals

- Do not change C4 indexer state runtime; TARGET 11.252 classified it as
  uninitialized/unconsumed.
- Do not change the C128 main-state storage/read surface unless source parity
  proves it is wrong.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf or low-precision work.
- Do not treat tiny BF16 differences as acceptable if they flip greedy tokens;
  greedy exactness remains the gate.

## Work Plan

### 1. Reproduce And Pin The SWA Owner

Using the tracked analyzer, reproduce the current first valid owner:

```bash
python debug/mtp/analyze_state_parity.py \
  --baseline performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json \
  --mtp performance_milestones/target11_mtp_online_c128_read_surface_port/raw/mtp_matrix_1_2_4_5_6_prefill_bank0.json \
  --output performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/raw/analysis_bs6_swa_owner.json \
  --batch-size 6
```

Confirm:

```text
first valid owner is swa.layer1 / commit_row_value_owner;
baseline trace 100 vs MTP trace 117;
uid0 position5 full_loc3077;
mapping aligned;
C128 raw-loc and C4 uninitialized-state owners remain filtered.
```

If the owner moves, explain why and follow the new first valid consumed owner
only if it is caused by better instrumentation rather than runtime changes.

### 2. Add SWA Row Lifecycle Instrumentation

Add or use trace fields to split the row lifecycle:

```text
target verify producer row before commit;
accepted commit source row;
accepted commit destination row;
row value immediately after copy;
row value after rejected-tail cleanup;
row value after snapshot/restore;
row value before next attention read;
baseline no-spec row at the same logical step.
```

For each observation record:

```text
uid / request index;
position;
full_loc;
swa_loc;
layer;
depth / draft row if applicable;
source row loc;
destination row loc;
checksum / abs_sum / short sample;
event name and trace index;
whether row is accepted, rejected, correction, or bonus.
```

### 3. Producer Versus Commit Split

Classify:

```text
producer mismatch:
    target-verify row differs from baseline before accepted commit.

copy source mismatch:
    accepted commit reads the wrong source row/depth/bank.

copy destination mismatch:
    accepted commit writes to the wrong destination loc.

copy value mismatch:
    source checksum is correct, dest mapping is correct, but post-copy value
    differs.

restore/cleanup mismatch:
    post-copy value is correct, then later changes.
```

Do not infer from final token drift.  Use row-level checksums and event order.

### 4. Source-Parity Audit

Compare Mini and SGLang accepted-row SWA commit semantics:

```text
which rows are verified;
which rows are accepted;
how accepted prefix lengths map to source depths;
how correction rows are written;
how SWA and full KV are copied;
when snapshot/restore happens;
how rejected draft rows are isolated;
how page/SWA locs are translated.
```

The report must explicitly answer:

```text
For uid0 position5 full_loc3077, what is the no-spec-equivalent source row?
Does Mini copy from that source row?
Does Mini write to the aligned destination row?
Is the row already different before commit?
```

### 5. Minimal Fix Policy

Allowed fixes:

```text
accepted-row source/depth selection;
SWA loc translation for accepted rows;
copy order for SWA/full/component rows;
snapshot/restore ordering if proven wrong;
trace/analyzer validity fields if needed.
```

Forbidden fixes:

```text
special-casing uid0/position5/layer1/full_loc3077/bs6;
changing logits/sampler;
disabling accepted commit;
discarding target verify;
relaxing greedy exactness;
undoing C128/C4 analyzer fixes without evidence.
```

### 6. Validation

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

Required checks:

```text
bs1/2/4/5/6 greedy exactness matrix;
MTP text sanity;
non-MTP baseline text sanity;
11.246 MoE microbatch focused guard if target verify row execution changes;
11.247 accepted-commit state/KV guard;
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

If only analyzer/trace instrumentation changes and no runtime behavior changes,
frozen artifacts may be reused, but the report must say exactness is unchanged.

## Stop Conditions

Stop and write the milestone report when one of these happens:

1. The SWA row value owner is classified into producer, copy source, copy
   destination, copy kernel value, restore/cleanup, or baseline alignment.
2. A minimal source-aligned fix makes bs1/2/4/5/6 exact.  Run all required
   guards and recommend the next MTP promotion target.
3. The current traces cannot split SWA producer/copy/restore.  Add small
   instrumentation if possible; otherwise write the precise next instrumentation
   target.
4. The first valid owner moves after better tracing.  Explain why and write the
   next target for the new owner.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/README.md
```

The report must include:

- final classification;
- SWA row lifecycle table for `uid0/position5/full_loc3077/layer1`;
- source-parity notes against SGLang;
- before/after first-owner ranking;
- any implementation changes;
- exactness matrix and smoke results if runtime changed;
- tests/static checks;
- remaining risks and next target recommendation.

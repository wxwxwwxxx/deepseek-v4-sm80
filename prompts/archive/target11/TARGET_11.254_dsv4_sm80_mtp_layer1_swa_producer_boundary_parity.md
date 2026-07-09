# TARGET 11.254: DSV4 SM80 MTP Layer1 SWA Producer Boundary Parity

## Status

Next after TARGET 11.253.

TARGET 11.253 classified the first valid consumed SWA owner as:

```text
classification: swa_target_verify_producer_value_owner
anchor: uid0 / position5 / full_loc3077 / swa.layer1
baseline row: trace100 baseline_after_normal_decode
MTP row: trace117 mtp_after_accepted_commit
```

Important conclusion from 11.253:

```text
The layer1 SWA row is already non-equivalent when target verify writes it.
Accepted-commit source/destination mapping is aligned.
Accepted-commit copy preserves the produced value.
Snapshot/restore and rejected-tail cleanup do not introduce the mismatch.
```

Therefore the next owner is upstream of the layer1 SWA store input, in the
target-verify producer path for the correction row.

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
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/
```

## Goal

Find the first producer boundary where target-verify correction row
`uid0/position5/full_loc3077` diverges from the no-spec normal decode row before
the layer1 SWA store input.

Split at least these boundaries:

```text
embedding / input row for the target token;
layer0 input;
layer0 output / residual after layer0;
layer1 input;
layer1 q/wkv projection input;
layer1 wkv_output;
layer1 kv_after_kv_norm_rope or equivalent normalized KV/RoPE row;
layer1 SWA store input.
```

The target passes with one of these classifications:

1. `producer_layer0_output_owner`: target-verify differs by the end of layer0.
2. `producer_layer1_input_owner`: layer0 output is equivalent, but layer1 input
   differs due to row/depth/batch materialization.
3. `producer_layer1_wkv_owner`: layer1 input is equivalent, but `wkv_output`
   differs.
4. `producer_layer1_kv_norm_rope_owner`: `wkv_output` is equivalent, but
   normalized/RoPE KV differs before store.
5. `producer_layer1_store_owner`: all upstream values are equivalent, but SWA
   store input/cache write differs.
6. `producer_metadata_owner`: values are equivalent when directly compared, but
   target verify uses wrong token, position, out_loc, full_loc, swa_loc, depth,
   or row metadata.
7. `producer_baseline_alignment_owner`: the baseline row chosen for comparison
   is not the no-spec-equivalent row.
8. `producer_instrumentation_no_go`: current hooks cannot compare the producer
   boundaries; add the smallest missing instrumentation or write a narrower
   instrumentation target.

If a minimal source-aligned fix is clear, it may be attempted.  Otherwise stop
with the precise owner and next repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/README.md
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/raw/
performance_milestones/target11_mtp_c4_indexer_state_validity_consume/README.md
performance_milestones/target11_mtp_online_c128_parity_planner_next_owner/README.md
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
prompts/TARGET_11.253_dsv4_sm80_mtp_swa_layer1_commit_row_value_parity.md
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
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/raw/analysis_bs6_swa_owner_layer1_lifecycle.json
performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/raw/mtp_matrix_1_2_4_5_6_swa_lifecycle_layer1.json
performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json
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
target-verify batch construction;
dsv4_target_verify_metadata;
row/depth and correction-row metadata;
positions / out_loc / full_loc / swa_loc;
target token used by verifier;
layer0 forward under target verify;
layer1 hidden-state input;
layer1 wkv and q/k/v normalize/RoPE path;
SWA store input and store_indexer/store_kv paths;
frozen-KV read-only flags and target verify temporary cache writes.
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
```

## Non-Goals

- Do not change accepted-commit copy/restore unless this target disproves
  TARGET 11.253.
- Do not change C4 indexer state runtime; TARGET 11.252 classified it as
  uninitialized/unconsumed.
- Do not change C128 main-state storage/read surface unless source parity proves
  it is wrong.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf or low-precision work.

## Work Plan

### 1. Reproduce The Producer Owner

Confirm the current owner with tracked analyzer:

```bash
python debug/mtp/analyze_state_parity.py \
  --baseline performance_milestones/target11_mtp_online_c128_read_surface_port/raw/baseline_matrix_1_2_4_5_6.json \
  --mtp performance_milestones/target11_mtp_swa_layer1_commit_row_value_parity/raw/mtp_matrix_1_2_4_5_6_swa_lifecycle_layer1.json \
  --output performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/raw/analysis_bs6_layer1_swa_owner.json \
  --batch-size 6
```

Confirm:

```text
swa.layer1 / commit_row_value_owner remains first valid owner;
TARGET 11.253 lifecycle still points to producer/store input as first mismatch;
accepted commit remains active and not fail-closed.
```

### 2. Add Focused Producer Boundary Trace

Add or use debug trace hooks for the same logical row in both no-spec baseline
and MTP target verify:

```text
request identity;
token id used for the row;
position;
full_loc;
swa_loc;
target-verify depth/category;
layer id;
boundary name;
shape/dtype;
checksum / abs_sum / sample;
rank-local or reduced/global status;
```

Required boundary labels:

```text
embedding_or_model_input
layer0_input
layer0_output
layer1_input
layer1_wkv_output
layer1_kv_norm_rope
layer1_swa_store_input
```

If exact internal names differ, use the closest existing code boundaries and
document the mapping.

### 3. Metadata Parity Check

Before trusting value checksums, verify metadata for the anchor:

```text
baseline target token for no-spec position5;
MTP verifier correction token / input token for depth0;
position equals 5;
full_loc equals 3077;
swa_loc equals 3077;
out_loc equals the intended cache row;
depth0 is correction row for accepted_prefix=0;
batch row and request mapping point to uid0;
```

If metadata differs, classify `producer_metadata_owner` and do not keep chasing
numeric values.

### 4. Boundary Bisection

For each boundary, compare baseline normal decode versus target verify:

```text
same metadata, same row, same tensor semantics;
rank-local if the operation is rank-local;
post-reduce/global if the operation uses communication;
finite values and shape-comparable checksums.
```

Stop at the first divergent boundary:

```text
layer0_output owner:
    layer0 input equivalent, layer0 output differs.

layer1_input owner:
    layer0 output equivalent, layer1 input differs.

wkv owner:
    layer1 input equivalent, wkv_output differs.

kv_norm_rope owner:
    wkv_output equivalent, normalized/RoPE KV differs.

store owner:
    normalized/RoPE KV equivalent, SWA store input/cache write differs.
```

Do not skip a divergent boundary just because the numeric delta is small.  If it
is sufficient to change greedy tokens downstream, it is correctness-relevant.

### 5. Source-Parity Audit

Compare Mini with SGLang for target verify producer construction:

```text
how verifier tokens are chosen;
whether correction row uses target token or draft token;
how hidden states are flattened/restored;
how row/depth order is represented;
how frozen KV is read;
how temporary SWA/full KV writes are isolated;
whether layer0/layer1 boundaries are run in normal-shape microbatches;
whether position/out_loc metadata matches no-spec normal decode.
```

The report must answer:

```text
For uid0 position5 full_loc3077, should target verify depth0 be numerically
equivalent to no-spec normal decode at each boundary?
If not, which contract says it may differ before final correction?
```

### 6. Minimal Fix Policy

Allowed fixes:

```text
target-verify token/position/out_loc metadata;
row/depth flatten/unflatten order;
hidden-state selection for correction rows;
normal-shape microbatching for a proven boundary;
temporary cache row routing if proven wrong;
trace/analyzer boundary semantics if needed.
```

Forbidden fixes:

```text
special-casing uid0/position5/layer1/full_loc3077/bs6;
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

1. The first producer boundary is classified with enough evidence for a repair
   target.
2. A minimal source-aligned fix makes bs1/2/4/5/6 exact.  Run all required
   guards and recommend the next MTP promotion target.
3. Metadata differs for the anchor.  Stop with `producer_metadata_owner` and
   write the smallest metadata repair target.
4. Current hooks cannot compare the required boundaries.  Add small
   instrumentation if possible; otherwise stop with a precise instrumentation
   target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer1_swa_producer_boundary_parity/README.md
```

The report must include:

- final classification;
- metadata parity table for `uid0/position5/full_loc3077`;
- boundary comparison table from layer0 through layer1 SWA store input;
- source-parity notes against SGLang;
- any implementation changes;
- exactness matrix and smoke results if runtime changed;
- tests/static checks;
- remaining risks and next target recommendation.

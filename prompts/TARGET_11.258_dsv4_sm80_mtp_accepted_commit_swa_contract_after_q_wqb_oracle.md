# TARGET 11.258: DSV4 SM80 MTP accepted-commit SWA contract after q_wqb oracle

## Status

Next after TARGET 11.257.

TARGET 11.257 classified the q_wqb row-shape line as:

```text
classification: q_wqb_contract_no_go
```

Important conclusion from 11.257:

```text
q_wqb row-shape sensitivity is real;
several q_wqb-only oracles make the carried q_wqb anchor exact;
no q_wqb-only contract makes the full greedy exactness matrix pass;
after the q_wqb target-normal-shape oracle, the next comparable owner is
swa.layer10 commit_row_value_owner after accepted commit.
```

Current visible owner after q_wqb normal-shape oracle:

```text
owner: commit_row_value_owner
component: swa.layer10
uid: 0
position: 5
full_loc: 3077
cached_len: 6
mapping_status: aligned
baseline_trace_index: 100
mtp_trace_index: 113
mtp_event: mtp_after_accepted_commit
baseline sha: ee6a68c388efa80d
mtp sha: ba9a1cc90877eb90
```

This target is contract-first.  The visible culprit has shifted across recent
targets as earlier owners were masked, disproved, or fixed by oracle gates.
Therefore do not patch `swa.layer10` locally just because it is the current
first comparable mismatch.  First define the accepted-commit SWA row lifecycle
contract, then prove which segment violates it.

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
```

Do not create new long-lived debug scripts only under `performance_milestones/`.
Milestone directories should contain reports, raw outputs, and one-off
artifacts.  If this target needs reusable accepted-commit/SWA lifecycle helpers,
put them under `debug/mtp/` and write outputs under:

```text
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/
```

## Contract To Establish

For greedy exact MTP, accepted-commit SWA state must satisfy this contract:

```text
For every accepted target-verify token T with logical identity
(uid, position, full_loc, depth), and for every SWA layer/component C:

1. Producer contract:
   target-verify forward computes the same C row value for T that no-spec
   greedy decode would compute for T under the chosen projection/oracle gates.

2. Source contract:
   before accepted commit, the target-verify temporary/source row for T is
   addressable by stable logical metadata and has not been overwritten,
   cleared, or remapped incorrectly.

3. Copy contract:
   accepted commit copies the exact source row bytes for T into the canonical
   committed SWA destination row for full_loc.

4. Destination contract:
   after accepted commit, future attention reads for full_loc resolve to the
   committed SWA destination row, not to stale target-verify scratch, a previous
   token, a tombstoned prefix handle, or a mismapped SWA slot.

5. Restore/cleanup contract:
   snapshot restore, rollback, temp-row cleanup, and lifecycle release must not
   destroy or replace committed SWA rows that were accepted.
```

This contract should be checked against SGLang where possible.  If SGLang uses
a different naming or storage layout, compare logical identity and lifecycle
events rather than raw buffer names.

## Goal

Classify the `swa.layer10` accepted-commit row value mismatch that appears
after the q_wqb normal-shape oracle, and decide whether it is the next true
contract violation or an artifact of the q_wqb oracle/debug path.

The target passes with one of these classifications:

1. `swa_commit_contract_producer_owner`: the SWA row is already wrong at
   target-verify layer10 producer/store input.
2. `swa_commit_contract_source_owner`: producer output is correct, but the
   source/temp row read by accepted commit is wrong, stale, overwritten, or
   misaddressed.
3. `swa_commit_contract_copy_owner`: source row is correct and destination
   mapping is correct, but accepted commit copies wrong bytes or incomplete
   bytes.
4. `swa_commit_contract_destination_mapping_owner`: accepted commit writes to
   or later reads from the wrong committed SWA/full/component slot.
5. `swa_commit_contract_restore_cleanup_owner`: the row is correct immediately
   after copy but becomes wrong after snapshot restore, rollback, cleanup, or
   lifecycle release.
6. `swa_commit_contract_q_wqb_oracle_artifact`: the mismatch exists only under
   q_wqb oracle gates and is not present on the default path under equivalent
   lifecycle tracing.
7. `swa_commit_contract_analyzer_owner`: the analyzer compares non-equivalent
   rows, stale rows, invalid rows, or incompatible layouts.
8. `swa_commit_contract_upstream_non_swa_owner`: SWA lifecycle is correct, but
   the row value differs because a still-earlier producer/value owner remains.
9. `swa_commit_contract_fix`: a generic contract fix lands and improves or
   passes the full exactness matrix.
10. `swa_commit_contract_instrumentation_no_go`: current hooks cannot split the
    lifecycle enough; add the smallest missing instrumentation or write a
    narrower target.

Do not branch on layer10, uid0, position5, full_loc3077, bs6, trace index, rank,
token, or prompt text.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/README.md
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/raw/state_trace_after_target_normal_shape_state_parity_analysis.json
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/raw/state_trace_baseline_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_q_wqb_cached_bf16_row_shape_contract/raw/state_trace_after_target_normal_shape_mtp_matrix_1_2_4_5_6.json
performance_milestones/target11_mtp_rank6_layer0_q_wqb_projection_parity/README.md
prompts/TARGET_11.257_dsv4_sm80_mtp_q_wqb_cached_bf16_row_shape_contract.md
prompts/TARGET_11.256_dsv4_sm80_mtp_rank6_layer0_q_wqb_projection_parity.md
prompts/TARGET_11.253_dsv4_sm80_mtp_swa_layer1_commit_row_value_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
debug/README.md
debug/mtp/README.md
```

Carry forward:

```text
Use the full 1/2/4/5/6 schedule, not isolated bs6.
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1.
Use q_wqb target-normal-shape only as an oracle/debug gate unless this target
proves it should be part of a broader exactness strategy.
Preserve TARGET 11.249/11.250 C128 main-state/read-surface behavior.
Preserve TARGET 11.251/11.252 analyzer validity rules.
Do not restore fail-closed accepted commit.
```

Important q_wqb oracle gates from 11.257:

```text
MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE=1
MINISGL_DSV4_MTP_Q_WQB_TARGET_FULL_ROWS=1
MINISGL_DSV4_MTP_Q_WQB_REFERENCE_GATE=1
MINISGL_DSV4_MTP_Q_WQB_GLOBAL_ROW_INVARIANT=1
```

Default runtime behavior should remain the main reference.  q_wqb gates are
debug oracles unless promoted by a later exactness decision.

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/utils/dsv4_mtp_debug.py
debug/mtp/analyze_state_parity.py
debug/mtp/run_matrix.py
```

Likely Mini code regions:

```text
target-verify accepted token construction;
accepted KV/component commit;
SWA temp/source row publication;
SWA committed row copy;
snapshot/restore and rollback cleanup;
prefix-handle/tombstone and lifecycle metadata;
DeepSeekV4KVCache SWA loc/full loc translation;
component state tracing and analyzer comparability rules.
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/mem_cache/
```

Use SGLang to define logical lifecycle behavior, not necessarily identical
buffer layout.  If SGLang commits accepted SWA/component rows through a mature
helper, document the equivalent Mini contract and consider adapting the
mechanism rather than inventing a new one.

## Non-Goals

- Do not optimize q_wqb further.
- Do not promote any q_wqb row-shape gate in this target.
- Do not change C4/C128 state lifecycle unless the SWA contract check proves
  shared accepted-commit metadata is wrong.
- Do not change MoE microbatching unless a guard proves it regressed.
- Do not patch logits/sampler.
- Do not disable target verify or accepted commit.
- Do not start graph/perf, communication-policy, PyNCCL, or low-precision work.

## Work Plan

### 1. Reproduce Both Views

Run or reuse full `1/2/4/5/6` traces for two views:

```text
A. default MTP path
B. q_wqb target-normal-shape oracle path
```

For each view, run `debug/mtp/analyze_state_parity.py` and identify:

```text
first comparable owner
component/layer
uid, position, full_loc
event name
source/destination mapping status
```

The target should explain whether `swa.layer10` is:

- already present on the default path but masked by q_wqb;
- introduced by the q_wqb oracle;
- or an analyzer/comparability artifact.

### 2. Define The Accepted-Commit SWA Event Trace

Add the smallest missing instrumentation needed to trace the carried logical row
through these checkpoints:

```text
target-verify producer/store input for swa.layer10;
target-verify temp/source row after store;
accepted-commit source row immediately before copy;
accepted-commit destination full/SWA/component loc before copy;
accepted-commit destination row immediately after copy;
destination row after snapshot restore / rollback / cleanup;
destination row at the next attention consume.
```

Each checkpoint should include:

```text
uid, position, full_loc, depth, layer, component;
source loc and destination loc;
SWA loc and full loc;
validity/ownership flag if available;
raw sha or checksum;
event id / trace index;
whether q_wqb oracle gates were active.
```

### 3. Compare Against Baseline Logical State

For the same logical row, compare no-spec baseline state at the equivalent
logical point:

```text
baseline decode store/commit of full_loc;
baseline committed SWA row after write;
baseline next consume if available.
```

The comparison must be logical-state based.  Do not require raw storage locs to
match when layouts differ, but require the row value and logical identity to be
comparable.

### 4. Split Producer vs Commit

Use the event trace to classify:

```text
if target-verify producer/store input differs from baseline:
  producer/upstream owner
elif source row before commit differs from producer:
  source publication/overwrite owner
elif destination after copy differs from source:
  copy owner
elif destination loc or next consume uses a different logical row:
  destination mapping owner
elif destination is correct after copy but wrong after restore/cleanup:
  restore/cleanup owner
else:
  analyzer/comparability owner
```

If producer is already wrong, do a narrow producer-boundary split for
`swa.layer10` only.  Do not start a broad layer bisection unless the target
cannot classify the first violated contract segment.

### 5. Minimal Generic Fix Policy

If the violated segment is clear and a generic fix is small, it may be
implemented.  Prefer fixes that:

```text
centralize accepted-commit ownership metadata;
copy by logical full/component loc rather than stale temp handle;
make validity/tombstone checks explicit;
align Mini's lifecycle with SGLang's logical contract;
preserve independent SWA lifecycle behavior.
```

Avoid fixes that:

```text
special-case layer10 or uid0;
hide the mismatch by skipping analyzer rows;
fall back to fail-closed accepted commit;
disable MTP accepted commits globally;
promote q_wqb gates as a workaround for SWA lifecycle bugs.
```

### 6. Validation

If runtime code changes, validate:

```text
full 1/2/4/5/6 exactness matrix;
text sanity smoke;
TARGET 11.246 MoE microbatch guard;
TARGET 11.249/11.250 C128 guards;
TARGET 11.251/11.252 analyzer validity guards;
TARGET 11.253 SWA commit/copy guard;
TARGET 11.256 q_wqb owner guard;
TARGET 11.257 q_wqb oracle evidence remains explainable.
```

Also run static checks:

```bash
python -m py_compile \
  debug/mtp/analyze_state_parity.py \
  debug/mtp/analyze_q_wqb_projection_parity.py \
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

1. The accepted-commit SWA lifecycle segment that violates the contract is
   classified with evidence.
2. A generic SWA accepted-commit contract fix lands and the full exactness
   matrix improves or passes.
3. The `swa.layer10` owner is proven to be a q_wqb oracle artifact, with the
   default-path first owner reported.
4. The analyzer is proven to compare invalid/non-equivalent rows, and the
   comparability rule is fixed or a precise follow-up is written.
5. Instrumentation is insufficient and the missing event/checkpoint is specified
   exactly.

Do not continue into graph/perf or broad operator bisection after the contract
segment is classified.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_accepted_commit_swa_contract_after_q_wqb_oracle/README.md
```

Include:

- final classification;
- explicit accepted-commit SWA contract summary;
- default-path versus q_wqb-oracle first-owner comparison;
- event trace table for `swa.layer10` or the actual first comparable owner;
- producer/source/copy/destination/restore/consume split;
- SGLang source-parity notes;
- any code changes and gates;
- before/after exactness matrix if a fix lands;
- commands and tests run;
- next target recommendation if exactness still fails.

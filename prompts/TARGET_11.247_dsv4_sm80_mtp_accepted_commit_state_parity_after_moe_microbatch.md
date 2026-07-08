# TARGET 11.247: DSV4 SM80 MTP Accepted-Commit State Parity After MoE Microbatch

## Status

Next after TARGET 11.246.

TARGET 11.246 implemented an opt-in runtime fix:

```text
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
```

The fix turns target-verify MoE into normal-shape-compatible microbatches:

```text
target verify rows: [B * W, hidden]
execute W chunks of [B, hidden]
reassemble to [B * W, hidden]
```

This closed the focused layer0 MoE owner:

```text
loc263/264/266/267 layer0.moe_output: 8/8 exact
loc263/264/266/267 layer0.post_moe_residual: 8/8 exact
focused bs2 visible output: exact
full matrix bs1/bs2/bs4/bs5: exact
```

The remaining failure is now only in the full `1/2/4/5/6` schedule:

```text
bs6 req4 baseline: [334,59275,8088,6073,344,260,14717,260]
bs6 req4 runtime:  [334,59275,8088,6073,344,260,4923,294]

bs6 req5 baseline: [17678,2067,3831,3955,361,582,2067,3362]
bs6 req5 runtime:  [17678,2067,3831,3955,361,582,77296,3362]
```

TARGET 11.246's owner-census notes are important:

```text
event13 req4 target verify agrees with current MTP-state normal oracle
event15 req5 target verify agrees with current MTP-state normal oracle
```

Therefore the current target-verify computation is self-consistent under the
MTP state.  The no-spec baseline has already diverged from the MTP committed
state before those rows are interpreted.

This target should find the first accepted/correction/bonus commit, cache state,
or request state component where MTP diverges from no-spec baseline after the
MoE microbatch fix.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the earliest event/state component where MTP committed state differs from
no-spec baseline in the full bs6 matrix, after the MoE microbatch runtime path
is enabled.

The target should answer:

```text
After MoE target-verify exactness is restored, which accepted/correction/bonus
commit first makes the future bs6 req4/req5 target state differ from no-spec
baseline?
```

The target passes when it produces one of these classifications:

1. `commit_row_value_owner`: a target-verify row selected for commit is already
   non-equivalent to the corresponding no-spec normal decode row.
2. `commit_mapping_owner`: the committed row value is correct, but it is copied
   to the wrong request, token position, full loc, SWA loc, C4/C128 loc, or
   component slot.
3. `component_state_owner`: hidden/KV value is correct, but an auxiliary DSV4
   component state such as SWA, C4, C128/compressed state, page table, or
   component loc map diverges.
4. `request_state_owner`: tensor state is correct, but seq len, req_to_token,
   accepted/correction/bonus accounting, or scheduler-visible request state
   diverges.
5. `attention_state_owner`: committed state is equal at high-level maps, but
   the next producer mismatch appears at an attention boundary such as
   `layer0.attention_wo_a_output`, with exact inputs but non-equivalent cache
   consumption.
6. `earlier_event_owner`: event13/event15 are downstream; an earlier target-
   verify commit or normal target decode step first diverges.
7. `instrumentation_no_go`: current hooks cannot compare no-spec and MTP state
   at the same event/row/position.

If a small source-aligned fix is clear, it may be attempted.  Otherwise close
with the owner and a narrower repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_moe_normal_shape_microbatch_runtime/README.md
performance_milestones/target11_mtp_target_verify_moe_normal_shape_microbatch_runtime/raw/
performance_milestones/target11_mtp_target_verify_layer0_moe_row_shape_precision_contract/README.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11.245_dsv4_sm80_mtp_target_verify_layer0_moe_row_shape_precision_contract.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 for this target unless
explicitly running a before/after comparison.
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not start graph/perf, CUDA graph capture, low-precision research, PyNCCL, or
communication-policy work.
Do not branch on batch size, request id, uid, event id, depth, rank, token,
layer, loc, expert, or prompt text.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/distributed/impl.py
```

Likely state/commit paths:

```text
target-verify accepted/correction/bonus row selection
accepted_kv_copied_tokens / target_commit_kv_copies
pre-verify snapshot
target-verify writes
committed snapshot
pre-restore
committed restore
req.complete_one()
req_to_token updates
SWA store/restore
C4/C128 compressed state writes and pending commits
component loc ownership maps
layer attention metadata generation after commit
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang as the reference contract for accepted/correction/bonus commit,
req_to_token updates, target-verify temporary writes, and DSV4 component state
publication.

## Non-Goals

- Do not disable accepted commit or fail closed to pass exactness.
- Do not patch MoE microbatch path except to fix a proven regression.
- Do not patch logits/sampler before proving committed state is exact.
- Do not start CUDA graph/perf work.
- Do not special-case bs6, req4, req5, event13, event15, locs, tokens, layers,
  ranks, or prompt text.
- Do not rely on isolated bs6; the failure must be studied in the full
  `1/2/4/5/6` schedule.

## Work Plan

### 1. Reproduce The Post-MoE State

Use the same shape as TARGET 11.246:

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
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
```

Confirm:

```text
focused bs2 remains exact
full matrix bs1/2/4/5 exact
full matrix bs6 req4/req5 still fail
event13/event15 target verify agrees with current MTP-state normal oracle
```

If bs6 no longer fails, rerun enough times to decide whether the failure was
unstable or fixed by unrelated changes, and document the new state.

### 2. Build Event Timeline And Commit Ledger

For the full `1/2/4/5/6` MTP run, write a timeline of target-verify and normal
target events leading to bs6 req4/req5 drift:

```text
event id
request ids / uids
input tokens
target tokens
draft tokens
accepted_prefix
mismatch_depth
correction / bonus rows
copy_rows
out_cache_loc
row_depths
seq_lens before/after
component locs: SWA, C4, C128/compressed, page table
```

Mark which rows are committed and which are rejected/temp-only.

### 3. No-Spec Baseline Alignment

For the same logical requests/tokens/positions in no-spec baseline and MTP:

```text
token id
position
request identity
full cache loc
SWA loc
C4/C128/component loc
seq len
```

Build a table mapping MTP committed rows to baseline normal decode rows.  If
mapping cannot be aligned, stop with `instrumentation_no_go` or
`commit_mapping_owner`.

### 4. Event-Level State Bisection

Find the earliest event where state diverges.  Start coarse:

```text
after each target-verify commit
after each normal target decode row
before event13
after event13
before event15
after event15
```

Compare compact hashes for:

```text
req_to_token slice for affected requests
seq_lens
full KV/component rows for committed tokens
SWA rows
C4/C128 compressed rows
component loc ownership maps
layer0/layer1/layer2 selected cache rows if cheap
```

Then narrow around the first divergent event.

### 5. Row-Value Versus Mapping Split

At the first divergent event, split:

```text
target-verify row hidden/state value before commit
normal baseline row hidden/state value
commit destination locs
cache value immediately after commit
cache value immediately before later read
request metadata after req.complete_one()
```

Classify:

```text
commit_row_value_owner:
    selected row is already non-equivalent before commit.

commit_mapping_owner:
    selected row is equivalent but written to wrong loc/request/depth.

component_state_owner:
    primary row value is equivalent but auxiliary component state diverges.

request_state_owner:
    tensor state is equivalent but scheduler/request metadata diverges.
```

### 6. Attention Boundary Guard

TARGET 11.246 noted that compact row-depth traces still report producer first
mismatches mostly at:

```text
layer0.attention_wo_a_output
```

If event-level state bisection points to an attention-state problem, split only
the needed attention boundary:

```text
layer0.input
q / q_norm / rope
consumed SWA/C4/C128/cache state
attention metadata
merged_attention_output_before_wo
wo_a output
wo_b output
final_attention_output
```

Do not redo MoE attribution unless `layer0.moe_input` or MoE guard regresses.

### 7. SGLang Source-Parity Table

Before any fix, write a compact table:

```text
Concept
SGLang behavior
Mini current behavior
Verdict / action
```

Cover:

```text
accepted/correction/bonus row commit
target-verify temporary row cleanup
req_to_token update order
seq_len update order
SWA/C4/C128 component state publication
page table / component loc ownership
normal target decode after accepted commit
```

### 8. Minimal Fix Policy

Allowed:

```text
source-aligned fix to commit the correct row/state
source-aligned fix to restore only rejected/temp rows
source-aligned fix to update req_to_token/seq_len/component locs in correct order
source-aligned fix to publish SWA/C4/C128 component state consistently
focused debug hooks for state bisection
```

Forbidden:

```text
branch on bs6, req4, req5, event13, event15, token, loc, uid, rank, layer, or
prompt text
disable accepted commit
overwrite final sampled token
copy no-spec baseline values into MTP state
turn off MoE microbatch to hide the owner
start graph/perf work
```

## Validation

Minimum static checks:

```bash
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/distributed/impl.py

git diff --check
```

Runtime checks:

```text
focused bs2 exactness with MoE microbatch enabled
full bs1/2/4/5/6 matrix with MoE microbatch enabled
bs6 full-matrix state bisection around first divergent event
accepted commit stats
MoE microbatch guard from TARGET 11.246 or equivalent
```

If a fix is attempted, rerun:

```text
focused bs2 MoE/SWA/layer2 guard
full bs1/2/4/5/6 matrix
bs6 event timeline after fix
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_accepted_commit_state_parity_after_moe_microbatch/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
event timeline and commit ledger
no-spec baseline alignment table
event-level state bisection
row-value versus mapping split
attention boundary guard if reached
SGLang source-parity table
first owner classification
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- No-spec baseline rows cannot be aligned with MTP committed rows.
- The earliest divergent event is found; close with that owner instead of
  continuing broad all-layer tracing.
- Current target-verify rows agree with MTP-state oracle but MTP committed state
  already differs from no-spec; do not patch the current event compute path.
- A proposed fix only works by branching on batch size, uid, request id, event,
  depth, rank, token, layer, loc, expert, or prompt text.
- A safe fix improves bs6 but regresses bs1/2/4/5, MoE microbatch exactness, or
  accepted commit accounting.

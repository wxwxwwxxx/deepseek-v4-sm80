# TARGET 11.20: DSV4 SM80 MTP Row-Depth Committed-State Baseline Parity

## Status

Next after TARGET 11.19.

TARGET 11.19 did not promote a runtime fix. It narrowed the current shared
failure to a row-depth committed-state lifecycle window:

```text
fresh 11.19 matrix:
  bs=1 pass
  bs=2 fail
  bs=4 fail
  bs=5 fail
  bs=6 fail

current shared visible failure for bs=4/5/6:
  req0 token6: baseline 582, MTP 223
```

The actionable anchor is:

```text
bs=4 uid0 event4:
  normal token before verify: 361
  target verify emits [582, 9628]
  copy_rows = 2
  depth0 row = accepted draft token 582
  depth1 row = correction token 9628

between event4 and event8:
  normal target emits 3362 and still appears visible-prefix correct

bs=4 uid0 event8:
  Mini current state emits correction 223
  baseline greedy under the same visible prefix should emit 582
```

TARGET 11.19's precise no-go:

```text
The first concrete unclosed owner is event4 committed depth1 correction row
state for token 9628. Existing traces cannot distinguish whether the
non-equivalent component is SWA/full KV, C4 compressed cache, C4 indexer
cache/state, online C128 partial state, component state, or next-step metadata.
```

Therefore TARGET 11.20 must add the missing baseline-side committed-state
oracle and compare row-depth state component by component.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the first concrete state component that differs between:

```text
Mini MTP committed state after bs=4 uid0 event4 depth0/depth1
```

and:

```text
baseline greedy state after emitting the same visible prefix and same row-depth
tokens.
```

The target passes when one of these is true:

1. It identifies and fixes the first concrete component mismatch, and the
   exactness matrix improves without regressing the scoped MoE fix.
2. Or it produces a precise no-go naming the first mismatching component and
   the SGLang lifecycle contract needed for the next implementation target.

The target should answer:

```text
When Mini commits target-verify depth0 token 582 and depth1 correction token
9628 at event4, are the rows written to every relevant state store identical
to the rows baseline greedy would have written after producing the same tokens?
```

If not, name the first mismatching store and row.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_accepted_commit_lifecycle_state_parity/README.md
performance_milestones/target11_mtp_post_moe_downstream_owner_census/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
prompts/TARGET_11.19_dsv4_sm80_mtp_accepted_commit_lifecycle_state_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Important TARGET 11.19 constraints:

```text
Do not fix event8 local operators first.
Do not fix attention_wo_b or indexer FP8 first.
Event8 row0 exactness under Mini current state only proves Mini self-consistency
after divergence, not baseline correctness.
Row category/count looked correct at event4, so the missing proof is state
equivalence for committed depth rows.
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
```

Likely Mini state stores:

```text
visible token ledger
req.device_len / cached_len / seq_len
req_to_token / page table / out locs
SWA/full KV rows
C4 compressed cache rows
C4 indexer cache/state
component loc / compressed component ownership, if present
online C128 pending banks
online C128 main/committed banks
target-verify metadata derived after commit
position ids and seq_lens for the next normal/verify call
scheduler active request ordering
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
target verify row category and accepted length handling
which target-verify hidden/KV rows are committed
how correction row state is written
online C128 pending/write/commit lifecycle
target KV view and frozen-KV draft metadata
metadata rebuild after accepted/correction commit
```

Use SGLang source behavior as the preferred contract. If Mini's direct-write /
snapshot / rollback model differs, prove equivalence by comparing state hashes
against baseline greedy.

## Non-Goals

- Do not start graph/perf work.
- Do not directly fix `attention_wo_b`, `attention_wo_a`, or indexer FP8 unless
  committed-state parity proves they are causal after event4 state is exact.
- Do not undo the TARGET 11.17 MoE row-invariant local fix.
- Do not disable accepted commit to pass exactness.
- Do not switch back to `legacy_target11_6`.
- Do not add parent batch size, request id, event id, row depth, rank, layer,
  token, or prompt-content special branches.
- Do not treat Mini-vs-Mini row0 exactness as sufficient. The required oracle is
  Mini committed state vs baseline greedy committed state under the same visible
  prefix.

## Work Plan

### 1. Reproduce The Current Shared Failure

Use the TARGET 11.19 contract:

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

Reproduce the matrix and, if still current, the shared failure:

```text
bs=4/5/6 req0 token6: baseline 582, MTP 223
```

If the matrix moves, follow the new first shared failure and explain why the
old event4 anchor is stale.

### 2. Add A Baseline-Equivalent Row-Depth Oracle

Build an opt-in debug path or script that can compare committed target-verify
rows against baseline greedy rows for the same visible prefix.

For the anchor:

```text
bs=4 uid0 event4 depth0 token 582
bs=4 uid0 event4 depth1 token 9628
bs=4 uid0 event8 pre-verify state after normal token 3362
```

Record baseline greedy state for:

```text
prefix before event4
prefix after event4 depth0 token 582
prefix after event4 depth1 token 9628
prefix after the next normal token 3362
```

Then compare Mini MTP committed state after the corresponding rows.

The oracle can be implemented by:

```text
running a baseline greedy replay with the same prompt/prefix and collecting
  compact state hashes at matching logical token positions;
or constructing a focused one-request replay after the same visible prefix;
or extending the existing exactness script to emit baseline state hashes for
  selected request/depth anchors.
```

Prefer compact hashes and metadata summaries over full tensor dumps.

### 3. Component Hash Ledger

For each anchor row/depth, record:

```text
logical token
logical position
physical out_loc / page loc
req_to_token row window
seq_len / cached_len / device_len
position ids used for the next call
metadata seq_lens and table indices
```

Hash and compare, where present:

```text
SWA/full KV row
C4 compressed cache row
C4 indexer cache/state row
component/compressed state row
online C128 pending bank row
online C128 committed/main bank row
any target-verify temp KV row that is later copied
next-step attention/indexer metadata derived from those rows
```

For each component, report:

```text
present in baseline?
present in MTP?
same logical position?
same physical location allowed or not?
same value hash?
max_delta if a value compare is cheap
first differing index if cheap
```

Physical locations may differ if the logical mapping is correct. Value hashes
and logical ownership are the important comparison.

### 4. Locate The First Mismatching Component

Follow this order:

```text
visible token ledger
row category / emitted token
copy_rows / committed row count
req_to_token logical mapping
SWA/full KV row value
C4 compressed cache row value
C4 indexer cache/state value
online C128 pending/main value
component/compressed state value
next-step metadata derived from committed state
```

Stop at the first mismatch that can explain event8 producing 223 instead of
baseline 582.

If all committed row values are baseline-equivalent, then event8's next-step
metadata or scheduler row mapping becomes the next owner. In that case, record
exact metadata fields and write the next target around metadata rebuild/order.

### 5. Source-Parity Table Against SGLang

Write a table before any fix:

```text
Concept
SGLang behavior
Mini baseline greedy
Mini MTP target-verify commit
Verdict / action
```

Cover at least:

```text
accepted/correction row compaction
which depth rows are committed
KV row writes for correction rows
online C128 pending begin/write/commit order
metadata rebuild after commit
seq_len and position semantics after multi-row commit
target KV frozen view vs Mini temp snapshot/restore model
```

### 6. Minimal Fix Policy

A minimal fix is allowed only after a concrete component mismatch is found.
Examples:

```text
depth1 correction row KV value differs -> fix row source or copy order
C4 compressed row missing -> commit compressed state for correction depth
C4 indexer state stale -> update/commit indexer state with correction row
online C128 pending not advanced -> fix pending/write/commit sequence
req_to_token logical mapping wrong -> fix mapping update
metadata built before commit is visible -> rebuild after commit
```

Validation after any fix:

```text
the exact component mismatch is closed
bs=4 uid0 event4->event8 anchor improves
full bs=1/2/4/5/6 matrix is rerun
TARGET 11.17 MoE pre-reduce sanity is not regressed
accepted commit stats are reported
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

Use the same six prompts from TARGET 11.19 unless the report states why a new
prompt set is required.

Focused gates:

```text
bs=4 uid0 event4 depth0/depth1 state hash comparison
bs=4 uid0 event8 pre-verify state hash comparison
bs=4 uid0 visible token ledger through token6
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_row_depth_committed_state_baseline_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
baseline-equivalent row-depth oracle description
component hash ledger
bs=4 uid0 event4 depth0/depth1 comparison
bs=4 uid0 event8 pre-verify comparison
SGLang source-parity table
first concrete component mismatch or precise no-go
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The matrix moves and the bs=4 uid0 event4->event8 anchor is no longer in the
  causal path.
- The baseline-equivalent state oracle cannot be built safely in this target;
  document exactly what instrumentation is missing and where it should be added.
- All committed row values are baseline-equivalent but event8 still diverges;
  then name next-step metadata/scheduler state as the next owner.
- A proposed fix passes only by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt content.
- A local operator drift is found but committed-state parity is not closed;
  defer the operator fix until the lifecycle state is proven equivalent.
- The matrix still fails after a component fix; close with the new first owner
  rather than starting graph/perf.

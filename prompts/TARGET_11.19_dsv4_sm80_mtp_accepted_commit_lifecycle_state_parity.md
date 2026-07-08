# TARGET 11.19: DSV4 SM80 MTP Accepted-Commit Lifecycle State Parity

## Status

Next after TARGET 11.18.

TARGET 11.18 did not promote a runtime fix. It showed that the remaining MTP
failures are not explained by a single direct `attention_wo_b` or indexer FP8
operator fix:

```text
post-11.17 matrix in TARGET 11.18:
  bs=1 pass
  bs=2 fail
  bs=4 fail
  bs=5 pass on that six-prompt matrix
  bs=6 fail
```

Key 11.18 conclusions:

```text
bs=2 event0 layer21 attention_wo_b_post_all_reduce_output:
  not a standalone wo_b bug;
  clean ranks have exact local wo_b and drift only after all-reduce;
  ranks2/3/4 already drift around attention_wo_a/local contribution.

bs=4 event2 layer32.indexer_query_fp8_values:
  real exact-input FP8 query value drift;
  downstream layer32 attention and MoE boundaries are exact in the same event;
  not yet proven causal for final visible token mismatch.

bs=4 event8 and bs=6 short trace:
  target-verify rows can be row0-exact under Mini's current committed state
  while the final visible sequence has already diverged from baseline greedy.
```

Therefore the highest-priority owner is accepted-commit lifecycle /
post-commit state parity, not a local operator patch.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the first accepted-commit lifecycle or post-commit state mismatch that lets
Mini's MTP path become internally self-consistent but different from baseline
greedy decoding.

The target passes when one of these is true:

1. It identifies and fixes a concrete lifecycle/state owner, and the exactness
   matrix improves without regressing the scoped MoE fix.
2. Or it proves a precise no-go with the first mismatching state component and
   an implementation plan for the next target.

The target should answer:

```text
After each target-verify event, do Mini's visible tokens, sequence lengths,
req_to_token rows, KV/cache/component rows, C128 pending/write/commit state,
indexer metadata, and scheduler request state match the baseline greedy state
that should exist after emitting the same visible tokens?
```

If the answer is no, name the first event and first component that diverges.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_post_moe_downstream_owner_census/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
prompts/TARGET_11.18_dsv4_sm80_mtp_post_moe_downstream_owner_census.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Anchor cases from TARGET 11.18:

```text
bs=4:
  event0 uid0 correction [64465], copy 1, exact
  event1 uid1 accept/correction [2353,1121], copy 2, mostly exact
  event2 uid2 accept/correction [80,361], copy 2, indexer FP8 strict drift but downstream exact
  event4 uid0 accept/correction [582,9628], copy 2, post-commit attention state drift
  event8 uid0 correction [223], copy 1, row0 exact under current Mini state
  final visible req0 already diverges from baseline at token6

bs=6:
  full matrix first visible mismatch req3 token1: baseline 10323, MTP 18
  short decode2 trace has initial target-verify rows exact under Mini state
  classify as lifecycle/state divergence until full state trace proves a local layer owner

bs=2:
  event0/event1 are in the causal window for req0 token4 mismatch
  layer21 attention_wo_a / local contribution is a secondary operator split,
  not the first target of this lifecycle target.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/mem_cache/
python/minisgl/attention/
performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py
```

Likely Mini lifecycle locations to inspect:

```text
target-verify temp KV snapshot/restore
accepted target-verify row selection
target_commit_kv_copies / accepted_kv_copied_tokens
accepted draft vs target correction vs bonus/tail row classification
visible token append and sequence length update
req_to_token / page table update
C4/C128 KV/component cache write/commit
online C128 pending/write/commit state
indexer metadata derived from committed state
scheduler request state after accept/reject/correction
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Relevant SGLang behavior to inspect and cite:

```text
frozen-KV target view and positions from seq_lens - 1
draft-token verification rows
accepted/rejected/correction row commit semantics
how target KV is written for accepted target-verify rows
online C128 pending/write/commit ownership
target-verify C128/indexer metadata preparation
what state is restored vs committed after verification
```

Use SGLang source behavior as the preferred contract. If Mini intentionally
differs, prove Mini's state after each event is equivalent to baseline greedy
for the same visible tokens.

## Non-Goals

- Do not start graph/perf work.
- Do not directly fix `indexer_query_fp8_values` unless lifecycle parity is
  proven and the indexer drift is proven causal.
- Do not directly fix `attention_wo_b_post_all_reduce_output`; TARGET 11.18
  showed it is collective propagation from earlier rank-local attention drift.
- Do not undo the TARGET 11.17 MoE row-invariant local fix.
- Do not disable accepted commit to pass exactness.
- Do not switch back to `legacy_target11_6`.
- Do not add parent batch size, active verify length, request slot, rank id,
  event id, layer id, token id, or prompt-content special branches.
- Do not accept row0 operator exactness alone as success if visible sequence or
  committed state differs from baseline greedy.

## Work Plan

### 1. Reproduce The Current Matrix And Events

Use the same contract as TARGET 11.18:

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

Reproduce at minimum:

```text
bs=1
bs=2
bs=4
bs=5
bs=6
```

If the exactness matrix moves again, follow the new traces and report the new
shape. Do not debug stale event numbers.

### 2. Define The Lifecycle State Ledger

Add opt-in debug instrumentation, or a post-processing script if raw data is
already available, that writes a compact event ledger for baseline and MTP.

For each request and event, record:

```text
request id / uid
event index
pre-event visible tokens
pre-event sequence length
draft tokens proposed
target-verify rows and row categories:
  accepted draft row
  target correction row
  bonus/tail row
  padding row
accepted draft tokens
rejected draft tokens
target correction token
bonus/tail token if any
post-event visible tokens
post-event sequence length
target_verify_rows_committed
target_commit_kv_copies
accepted_kv_copied_tokens
fail_closed and blocker
```

Also record compact hashes/checksums for state components:

```text
req_to_token row for each active request
KV page ids / offsets touched
C4/C128 cache rows touched
component loc / compressed state ownership if present
online C128 pending/write/commit buffers
indexer metadata derived from committed state
position ids / seq_lens used for next target verify
scheduler active request ordering
```

Keep this debug path opt-in. Avoid dumping full tensors unless a focused owner
requires it; the 11.18 raw trace was already very large.

### 3. Build Baseline-Equivalent State Checks

The main comparison is not only MTP-vs-MTP row0 exactness. After each event,
compare MTP committed state against the baseline greedy state after the same
visible tokens.

For each request:

```text
if MTP visible tokens == baseline prefix of same length:
  compare sequence length and req_to_token mapping
  compare relevant KV/component/cache rows by hash
  compare next-step metadata inputs
else:
  identify the event that first produced the divergent visible token
```

If the baseline greedy sequence and MTP visible sequence diverge, stop treating
later exact target-verify rows under Mini state as proof of correctness. They
are only proof that Mini is self-consistent after divergence.

### 4. Focused Anchors

#### bs=4 event4 -> event8

Trace uid0 from event0 through event8:

```text
event0 correction [64465]
event4 accept/correction [582,9628]
event8 correction [223]
final visible req0 mismatch at token6
```

Find the first of:

```text
visible token ledger mismatch
target row category mismatch
wrong row committed
wrong number of rows committed
KV/cache row missing or stale
C128 pending/write/commit mismatch
req_to_token/page table mismatch
position/seq_len mismatch for next verification
metadata derived from stale state
```

#### bs=6 initial-request state

Use a cheap short trace first, then extend only as needed:

```text
full matrix first visible mismatch req3 token1: baseline 10323, MTP 18
short trace event0-5 row0 parity exact under Mini state
```

Determine whether the mismatch is caused by:

```text
visible correction token chosen differently
wrong baseline comparison alignment
wrong request/row mapping
accepted commit state written to the wrong request
next-step metadata/position mismatch
```

#### bs=2 secondary attention split

Keep this secondary. Use only if lifecycle ledger does not find an earlier
owner:

```text
event0 layer21 merged_attention_output_before_wo
inverse RoPE
attention_wo_a_output on ranks2/3/4
local wo_b contribution
post-all-reduce propagation
```

This split should explain whether bs=2 has an independent rank-local attention
operator owner or is also post-commit state/metadata drift.

### 5. Source-Parity Table Against SGLang

Write a source-parity table before promoting any fix:

```text
Concept
SGLang behavior
Mini baseline greedy
Mini MTP target-verify
Verdict / action
```

Cover at least:

```text
frozen target KV view
target-verify row category mapping
accepted/rejected/correction/bonus token emission
which target-verify rows are committed
KV/cache rows written for accepted rows
online C128 pending/write/commit ownership
req_to_token/page-table updates
seq_lens / positions for the next verify call
metadata rebuild after commit
```

If SGLang has a mature implementation for a contract Mini lacks, prefer
adapting that contract over inventing a mini-only state protocol.

### 6. Minimal Fix Policy

A minimal fix is allowed if and only if the ledger identifies a clear owner.
Examples:

```text
commit the wrong row category -> fix row selection
copy rows but not req_to_token -> fix mapping update
commit C4/C128 KV but not pending compressed state -> fix pending commit
restore temp state after commit in the wrong order -> fix restore/commit order
metadata built before committed state is visible -> rebuild after commit
seq_len advanced by draft rows instead of visible rows -> fix seq_len update
```

Validation must show:

```text
the first state mismatch is closed
the same anchor case improves
MoE pre-reduce sanity from TARGET 11.17 still passes or is not regressed
full bs=1/2/4/5/6 matrix is rerun
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

Use the same six fixed prompts from TARGET 11.18 unless the report clearly
states why a new prompt set is needed.

Focused gates:

```text
bs=4 event4->event8 lifecycle ledger
bs=6 initial-request lifecycle ledger
bs=2 event0 attention split only if lifecycle ledger does not close the owner
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_accepted_commit_lifecycle_state_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
event lifecycle ledger
baseline-equivalent state comparison
bs=4 event4->event8 anchor analysis
bs=6 initial-request state analysis
bs=2 attention split if used
SGLang source-parity table
first lifecycle/state owner or precise no-go
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The post-11.18 matrix no longer reproduces and the first owner moved.
- Later target-verify rows are exact only under an already-diverged Mini state;
  in that case name the earlier divergence event instead of fixing later
  operators.
- The first mismatch is a SGLang lifecycle contract Mini does not implement and
  the fix is larger than one focused target; document the contract and next
  implementation plan.
- A proposed fix passes only by branching on batch size, request id, event id,
  rank, layer, token, or prompt content.
- A local operator drift is found but its downstream is exact and it is not
  causal for visible-token divergence; defer it until lifecycle parity is
  closed.
- The exactness matrix still fails after a lifecycle fix; close with the new
  first owner rather than starting graph/perf.

# TARGET 11.296: DSV4 SM80 MTP Row0 Logits Parity After Accepted Commit

## Status

Next after TARGET 11.295.

TARGET 11.295 successfully ported the online C128 MTP pending/write/commit
lifecycle enough that accepted commit no longer fail-closes:

```text
accepted_kv_commit_fail_closed = False
accepted_kv_commit_blocker = ""
accepted_kv_copied_tokens = 4
draft_tokens_accepted = 1
```

However, greedy exactness still fails:

```text
baseline = [11111, 64465, 361, 582, 9628, 3362, 582, 18]
mtp      = [11111, 64465, 361, 582, 9628, 3362, 223, 18]
```

Do not work on CUDA graph, throughput, or serving datasets in this target.

## Goal

Find and fix why flattened target-verify row0 logits after an accepted commit do
not match normal target decode logits for the same prefix/token/position.

The suspected failing comparison from TARGET 11.295 is:

```text
committed_seq_len = 10
input_tokens = [3362, 582]
positions = [10, 11]
target_verify row0 token = 223
baseline normal decode token after 3362 = 582
```

This is not a normal MTP draft miss.  MTP draft tokens may be wrong, but target
verify row0 must match baseline target decode when it sees the same committed
prefix, same input token, same position, and same target KV/component state.

## Speculative Stats Glossary

Use these meanings in code comments, reports, and debug output.  If current
stat names are ambiguous, clarify them in the report and optionally add
better-named aliases.

```text
draft_tokens_proposed:
    tokens produced by the MTP draft path.

draft_tokens_verified:
    draft tokens compared against target-verify outputs.

draft_tokens_accepted:
    draft tokens whose value matched target verify and became visible output.

draft_tokens_rejected:
    draft tokens whose value did not match target verify.

target_correction_tokens:
    target-model tokens emitted at the first rejection point.

target_verify_rows:
    rows computed by the target verify forward.  These can include accepted
    draft rows, correction rows, and bonus/tail rows.

target_verify_rows_committed:
    target-verify rows made visible in target KV/component/state.

accepted_draft_rows_committed:
    committed rows corresponding to accepted draft tokens.

correction_rows_committed:
    committed rows corresponding to target correction tokens.

accepted_kv_copied_tokens:
    historical mini stat name for committed target-verify rows.  Do not read it
    as "accepted draft tokens"; it may include correction or bookkeeping rows.
```

Required invariant for greedy speculative decoding:

```text
baseline greedy output == MTP speculative greedy output
```

It is normal for:

```text
draft token != target verify token
```

It is a bug if, under identical prefix/state:

```text
normal target decode row0 logits != target-verify row0 logits
```

## Starting Evidence From TARGET 11.295

Important trace:

```text
debug_trace[1]:
  input_tokens = [361, 582, 671]
  target_tokens = [582, 9628, 6102]
  draft_tokens = [582, 671]
  accepted_prefix = 1
  copy_rows = 2
  emitted_tail = [582, 9628]

debug_trace[2]:
  committed_seq_len = 10
  input_tokens = [3362, 582]
  positions = [10, 11]
  target_tokens = [223, 18]
  draft_tokens = [582, 19]
  accepted_prefix = 0
  copy_rows = 1
```

Also:

```text
c128_out_loc = []
sequence length < 128
c128_pending_write_commit = "ready"
```

This suggests the current first visible drift is not caused by a C128 boundary
write.  It may be caused by the previously committed raw/SWA/C4/indexer row,
request/token table state, hidden state, or target-verify metadata.

## Work Plan

### 1. Preserve controls

Keep these controls intact:

- baseline normal greedy decode;
- rollback-only MTP exact path;
- accepted-commit MTP failing path from TARGET 11.295.

Do not hide the mismatch by disabling commit, recomputing accepted tokens
sequentially, or weakening greedy exactness.

### 2. Rule out trivial nondeterminism

Before deep bisection, add a lightweight determinism gate.  This should not
become a large standalone target.

Run the same short prompt multiple times with CUDA graph disabled and PyNCCL
disabled:

- baseline normal decode, 3-5 repeats;
- MTP accepted-commit path, 3-5 repeats.

Record whether token ids are stable.  Also dump row0 top-k logits and margins:

```text
top10 token ids/logits for normal decode row0
top10 token ids/logits for target-verify row0
logits[baseline_top1]
logits[mtp_top1]
top1 - top2 margin for each path
max_abs_diff between comparable logits if available
```

Interpretation:

```text
More likely nondeterminism:
    baseline or MTP token ids vary across repeats, or top1/top2 margins are
    extremely small.

More likely state/metadata bug:
    baseline is stable, MTP is stable, but they stably disagree; margins are not
    tiny; metadata/KV/state hashes differ.
```

If the evidence points to numerical sensitivity only, report it and propose a
small numerical-stability follow-up.  Otherwise continue with row0 parity and
state-owner bisection.

### 3. Build a row0 parity harness

Add a focused debug mode or script that captures, for the same request:

```text
normal target decode:
    prefix through token 3362
    next input/position
    logits/top-k for the next token

MTP target verify:
    same committed prefix
    row0 input token/position
    row0 logits/top-k
```

Record:

- top-10 token ids and logits for both paths;
- max absolute / relative logit difference;
- first layer where hidden states diverge, if practical;
- target input token, position, and output token;
- committed prefix token ids.

### 4. Audit target-verify row0 metadata

For the mismatch-producing verify batch, dump and compare:

- `raw_out_loc`;
- `out_cache_loc`;
- `positions`;
- `seq_lens`;
- `req_seq_lens`;
- `extend_lens`;
- `swa_page_indices`;
- `swa_topk_lengths`;
- `c4_out_loc`;
- `c4_page_indices/full_indices`;
- `c4_indexer` metadata;
- `c128_out_loc` and C128 lifecycle status;
- `req_to_token` / page-table slice around positions 7-11.

The row0 metadata should describe the same target state as normal decode for
the same prefix.  Any intentional difference must be justified with a SGLang
reference.

### 5. Audit the previous accepted commit

The mismatch appears after the previous verify committed rows.  Compare the
rows committed by the previous verify with normal sequential target decode:

```text
position 7 / token 361
position 8 / token 582
correction or bonus row if committed
```

For each relevant layer/component, compare hashes or sampled values:

- raw/full KV mapping;
- SWA KV;
- C4 compressed KV;
- C4 indexer KV/cache;
- C4 compression state;
- C128 state only if touched;
- hidden states before/after norm if available.

Find the first owner that differs.  If this becomes too large, start with layer
0 and the first DSV4 attention/cache component.

### 6. Fix the smallest proven owner

Examples of plausible fixes:

- commit only accepted draft rows plus exactly one correction row, not an
  ambiguous bonus/tail row;
- fix `req.device_len`, page-table, or `req_to_token` update order after commit;
- fix `out_cache_loc`/position mapping for target-verify row0;
- fix SWA/C4/indexer row copy direction or row selection;
- add a target-verify-specific metadata path for a component still using decode
  assumptions.

Do not make broad refactors unless the first-mismatch evidence requires them.

## Verification Gates

Minimum:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_row0_logits_parity_after_commit/raw/exactness_baseline_bs1_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --draft-len 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_row0_logits_parity_after_commit/raw/exactness_mtp_commit_draft2_bs1_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --enable-spec --draft-len 2
```

Required to unblock TARGET 11.3:

```text
accepted_kv_commit_blocker is empty/None
draft_tokens_accepted > 0
target_verify_rows_committed > 0 or accepted_kv_copied_tokens > 0
baseline greedy token ids == MTP token ids
row0 normal-decode logits == target-verify row0 logits within normal numerical tolerance
```

Then extend only to:

```text
bs=1/2/4, draft_len=2
```

Do not run macro performance in this target.

## Stop Lines

Stop and report if:

- row0 parity cannot be captured with enough detail to identify a first owner;
- exactness can only be restored by disabling accepted commit;
- exactness requires recomputing all accepted tokens through normal sequential
  target decode;
- the first mismatch owner is a broad SGLang target-verify mechanism that needs
  a separate porting target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_row0_logits_parity_after_commit/README.md
```

Include:

- stats glossary/invariant used in the report;
- nondeterminism repeat results and top-k margin check;
- exact repro commands;
- baseline vs MTP token ids;
- row0 logits/top-k comparison;
- metadata comparison;
- previous accepted-commit row comparison;
- first mismatch owner;
- fix summary or next narrow target;
- whether TARGET 11.3 graph/perf is unblocked.

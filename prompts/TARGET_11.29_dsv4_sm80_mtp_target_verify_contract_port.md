# TARGET 11.29: DSV4 SM80 MTP Target-Verify Contract Port

## Status

Next after TARGET 11.28.

TARGET 11.28 found a source-level no-go for accepted-KV commit in the current
mini MTP runtime.  The rollback-only path is exact, but accepted verify rows are
not safe to commit because mini does not yet build SGLang-equivalent
target-verify state.

Do not work on CUDA graph or throughput in this target.  The only goal is to
make accepted target-verify commit exact, or fail closed with a precise
SGLang-parity blocker.

## Starting Evidence

TARGET 11.28 identified these source-level mismatches:

- mini builds flattened verify as a generic `Batch(phase="decode")`, not as a
  target-verify metadata mode with fixed `speculative_num_draft_tokens` rows per
  request;
- mini lacks the SGLang `OnlineC128MTPController` pending/write/commit lifecycle;
- the active top-k 1 acceptance loop compares the first draft row and then
  breaks on `target == draft` without marking that row accepted, emitting it, or
  selecting its KV/component rows for commit;
- historical accepted-commit attempts changed greedy output:
  `[11111, 64465, 361, 582, 9628, 3362, 223, 18]` instead of
  `[11111, 64465, 361, 582, 9628, 3362, 582, 18]`;
- current rollback-only MTP remains exact only because
  `accepted_kv_copied_tokens=0`.

Therefore the owner is not a single layer kernel.  The owner is the
target-verify state contract before and around the layer loop.

## Goal

Implement the minimum mini-sglang target-verify contract needed for exact
accepted-KV commit:

```text
baseline greedy output
  ==
MTP flattened target verify + accepted front-chain commit output
```

First passing case:

```text
TP8 real /models/DeepSeek-V4-Flash
page_size=256
bs=1
draft_len=2
decode_len=8
CUDA graph disabled
at least one accepted draft token committed
```

Then extend to bs=2/4 and draft_len=4 only after the bs=1 first-accepted-row
case is exact.

## SGLang References

Use these as the source contract:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_worker_v2.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Key behaviors to preserve or explicitly fail closed:

- draft attention reads frozen target KV with rope phase `seq_lens - 1`;
- target verify advances over the draft-token extension;
- target-verify metadata uses fixed `speculative_num_draft_tokens` rows per
  request;
- top-k 1 draft tokens are a front chain, so accepted prefix rows are selected
  directly and rejected tails remain invisible;
- C128 online compression state uses pending/write/commit semantics, not a raw
  snapshot-copy shortcut.

## Required Work

### 1. Add explicit target-verify metadata mode

Do not reuse the generic decode metadata path blindly.

Add a mini target-verify owner path that can express the SGLang contract:

```text
speculative_num_draft_tokens = draft_len or active verify length
seq_lens = committed_seq_lens + speculative_num_draft_tokens
extend_lens = [speculative_num_draft_tokens] * batch_size
num_tokens = speculative_num_draft_tokens * batch_size
positions = committed positions through the verify extension
out_cache_loc = flattened/front-chain verify row locations
```

Record exactly where this differs from current
`_make_mtp_flattened_verify_batch()` and
`DSV4AttentionBackend._build_metadata()`.

If a reduced active verify length is used after an early mismatch, prove that it
is still SGLang-equivalent.  Otherwise keep the fixed draft-token count for
metadata and only select accepted rows at commit time.

### 2. Fix top-k 1 front-chain acceptance bookkeeping

Fix the active acceptance loop so `target == draft` does not just break.

It must:

- mark the draft row accepted;
- append the accepted token to the emitted token stream;
- include the corresponding front-chain verify row in the commit set;
- keep rejected tail rows invisible;
- emit the target correction token when the first mismatch occurs;
- update stats so `draft_tokens_accepted`,
  `accepted_kv_copied_tokens`, and `target_commit_kv_copies` reflect real
  accepted work.

The target is not passing if it remains exact only because accepted commit is
zero.

### 3. Handle C128 MTP pending/write/commit semantics

Map SGLang's `OnlineC128MTPController` to mini's DSV4 C128 state.

Preferred path:

- port the minimum pending/write/commit lifecycle needed for top-k 1 MTP target
  verify;
- commit pending state only when the accepted front-chain path is committed;
- discard or keep invisible rejected pending state.

Allowed oracle path:

- temporarily disable C128 MTP commit or force a narrow diagnostic mode to prove
  whether C128 is the first remaining blocker.

Fail-closed path:

- if C128 pending/write/commit is required but too large to port in this target,
  stop and report that accepted-KV commit cannot be enabled for DSV4 until C128
  MTP state ownership is implemented.

Do not silently copy partial C128 rows/state and call it exact.

### 4. Preserve rollback-only exactness

Keep the TARGET 11.27 rollback-only path as a control.

Before enabling any commit variant, verify that rollback-only still matches
baseline greedy output for:

- bs=1/2/4;
- draft_len=2;
- short deterministic decode.

### 5. Add focused diagnostics

For the first passing or failing bs=1 case, log:

- verify input tokens and positions;
- `seq_lens`, `extend_lens`, and `out_cache_loc`;
- accepted prefix length and commit row mapping;
- raw/full KV commit rows;
- SWA, C4, C128, and indexer commit rows or explicit not-applicable status;
- C128 pending/write/commit lifecycle events;
- rejected-tail isolation checks.

Keep logs small enough to inspect quickly.

## Verification Gates

Run real TP8 on A100/sm80:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_target_verify_contract_port/raw/exactness_baseline_bs1_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_target_verify_contract_port/raw/exactness_mtp_commit_draft2_bs1_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --enable-spec --draft-len 2
```

Adjust the script only if it lacks a flag needed to force accepted commit or
emit the new diagnostics.  Keep graph disabled.

Required pass matrix:

1. rollback-only control remains exact;
2. bs=1/draft_len=2 exact with `accepted_kv_copied_tokens > 0`;
3. bs=1/draft_len=2 rejected tails remain isolated;
4. bs=2/4 exact after bs=1 passes;
5. draft_len=4 exact or documented as a smaller follow-up blocker.

## Stop Lines

Stop this target and report no-go if:

- accepted commit still changes greedy output after metadata and acceptance
  bookkeeping match SGLang;
- exactness requires recomputing every accepted token through normal sequential
  target decode;
- C128 pending/write/commit must be ported but cannot be done safely within this
  target;
- the fix only passes by keeping `accepted_kv_copied_tokens=0`.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_contract_port/README.md
```

Include:

- a SGLang-vs-mini source parity table after the implementation;
- exact command lines;
- exactness matrix before and after;
- accepted/verified/rejected token stats;
- whether C128 pending/write/commit was ported, disabled as an oracle, or is the
  remaining blocker;
- whether TARGET 11.27 should be rerun;
- whether TARGET 11.3 graph/perf work is unblocked.

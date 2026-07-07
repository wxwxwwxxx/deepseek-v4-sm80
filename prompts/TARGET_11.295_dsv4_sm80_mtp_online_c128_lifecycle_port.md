# TARGET 11.295: DSV4 SM80 MTP Online C128 Lifecycle Port

## Status

Next after TARGET 11.29.

TARGET 11.29 ported the target-verify metadata owner and top-k 1 front-chain
acceptance bookkeeping, but it correctly failed closed on accepted-KV commit:

```text
accepted_kv_copied_tokens = 0
accepted_kv_commit_blocker = c128_online_mtp_pending_write_commit_not_ported
```

This target owns that blocker.  Do not work on CUDA graph, macro throughput, or
dataset acceptance yet.

## Goal

Port or safely reimplement the minimum SGLang-equivalent online C128 MTP
pending/write/commit lifecycle needed for exact DSV4 accepted target-verify
commit in mini-sglang.

First success gate:

```text
TP8 real /models/DeepSeek-V4-Flash
page_size=256
bs=1
draft_len=2
decode_len=8
CUDA graph disabled
accepted_kv_copied_tokens > 0
baseline greedy output == MTP accepted-commit output
```

If C128 cannot be ported safely in this target, keep accepted commit fail-closed
and report the exact missing API/kernel/ownership piece.

## Important Caution

TARGET 11.29 proves C128 is a source-level blocker to safe commit; it does not
yet prove that every historical commit drift was caused only by C128.  This
target should therefore include a narrow C128 oracle or diagnostic check where
possible.

The desired outcome is either:

- accepted commit becomes exact after the C128 lifecycle is ported; or
- C128 is cleared as one blocker and the next first-mismatch owner is identified.

Do not silently enable accepted commit by copying partial C128 rows/state.

## Speculative Stats Note

Use the TARGET 11 stats glossary.  In particular:

```text
draft_tokens_accepted != accepted_kv_copied_tokens
```

`draft_tokens_accepted` counts accepted draft tokens.  `accepted_kv_copied_tokens`
is a historical mini stat for committed target-verify rows and may include
correction or bookkeeping rows.  Reports from this target should spell out which
rows were committed: accepted draft rows, correction rows, and any bonus/tail
rows.

## SGLang References

Use SGLang as the source contract:

```text
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_worker_v2.py
```

Map these exact behaviors:

- `OnlineC128MTPController.prepare_forward()` commits previous pending state,
  then marks new pending state for target verify.
- `mark_pending` records pending target-verify sequence lengths by request pool
  index.
- `write_prefix_states` writes ratio-128 verify prefix states into temporary
  online MTP state slots.
- `commit_pending` publishes the accepted pending state into the main C128
  compression-state pool.
- Rejected tail states remain invisible or are discarded.

## Mini Starting Points

Relevant mini files:

```text
python/minisgl/engine/engine.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
```

Current mini state:

- C128 cache and C128 state pools already exist.
- MTP target-verify metadata and acceptance candidate bookkeeping exist.
- Accepted commit is fail-closed by `_mtp_accepted_commit_blocker()`.
- Missing pieces include pending seq-lens storage, online MTP state-slot
  offset/max-draft-token APIs, C128 write-prefix/commit kernels, and integration
  around target verify.

## Work Plan

### 1. Source-parity map

Before coding broad changes, write a short parity table:

```text
SGLang owner/API/kernel
mini equivalent
same / missing / different
planned fix
```

Cover:

- pending seq-lens buffer;
- max requests and request indices;
- max draft tokens;
- online MTP state-slot offset;
- full-token to SWA mapping;
- C128 state buffer layout and stride;
- C128 compressor head dim and APE layout;
- write-prefix kernel inputs;
- commit-pending kernel inputs;
- lifecycle call sites.

### 2. Add mini KV/cache APIs

Add the smallest auditable APIs needed by the lifecycle:

```text
get_online_c128_mtp_pending_seq_lens()
get_online_c128_mtp_state_slot_offset()
get_online_c128_mtp_max_draft_tokens()
full_to_swa mapping accessor if needed
C128 state pool accessor/metadata if needed
```

The extra online MTP state slots must not collide with normal C128 state slots.
Record the memory overhead in bytes and equivalent KV-token capacity.

### 3. Port or bind the C128 kernels

Preferred route:

- port the SGLang CUDA kernels into mini's existing extension/JIT style;
- keep names and semantics close to SGLang;
- add a small unit/micro smoke that does not load model weights.

Minimum kernels:

```text
mark_pending
write_prefix_states
commit_pending
```

If direct TVM/tvm.ffi binding is too expensive in mini, use a mini-owned CUDA
extension or JIT wrapper.  Do not fall back to a slow Python/Torch implementation
for the final path, except as a correctness oracle.

### 4. Add a mini OnlineC128MTP owner

Implement a small controller or owner object in mini that mirrors SGLang's
lifecycle:

```text
prepare_forward(non-target-verify):
    commit previous pending, then return 0

prepare_forward(target-verify):
    commit previous pending
    mark pending for current verify requests
    return online_state_slot_offset

write_prefix_states(layer_id, compressor, kv_score_input):
    write target-verify prefix states into online MTP slots

commit_pending(current reqs/seq_lens):
    publish accepted pending state into main C128 state

clear():
    discard pending context when verify is abandoned or batch is idle
```

The owner must make rejected tails invisible.  If accepted prefix length is less
than the full verify length, commit only the accepted front-chain state.

### 5. Wire into target verify

Integrate with TARGET 11.29's target-verify path:

- mark pending before C128 target-verify write-prefix state is produced;
- pass online state-slot offset into C128 metadata/state writes;
- commit pending only after accepted prefix is selected;
- leave rollback-only exactness intact;
- keep commit disabled if any required C128 lifecycle event is missing.

### 6. Add narrow diagnostics/oracles

Add diagnostics for the first bs=1 case:

- pending seq lens before/after mark;
- online state slot offset;
- number of write-prefix calls and layers;
- number of commit-pending calls and accepted rows;
- accepted prefix length;
- C128 state locs touched;
- whether the first mismatch, if any, appears before or after C128 commit.

Where useful, add a temporary oracle path:

- compare C128 state after accepted commit with sequential target decode for one
  request/layer/chunk;
- or run a debug mode that disables C128 accepted-state publication to prove the
  drift owner.

Oracle/debug paths must remain opt-in and must not be promoted as performance
paths.

## Verification Gates

Run in this order.

### A. Static and micro

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py

python -m compileall -q python/minisgl
```

Add and run a no-weight micro smoke for the C128 lifecycle if practical:

```text
mark_pending -> write_prefix_states -> commit_pending
```

The micro should check tensor shapes, pending buffer values, state-slot writes,
and commit output for a tiny synthetic request.

### B. Rollback control

Before accepted commit:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_online_c128_lifecycle_port/raw/exactness_mtp_rollback_draft2_bs124_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --disable-pynccl \
  --enable-spec --draft-len 2
```

Rollback-only exactness must still pass.

### C. Accepted commit

Run the smallest real gate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py \
  --output performance_milestones/target11_mtp_online_c128_lifecycle_port/raw/exactness_mtp_commit_draft2_bs1_tp8.json \
  --page-size 256 --num-pages 16 --decode-len 8 --batch-size 1 \
  --disable-pynccl --enable-spec --draft-len 2
```

Required:

```text
accepted_kv_copied_tokens > 0
draft_tokens_accepted > 0
baseline greedy token ids == MTP token ids
rejected_tail_isolation_checks > 0
accepted_kv_commit_blocker is empty/None
```

Then extend to:

```text
bs=1/2/4, draft_len=2
bs=1/2/4, draft_len=4 if draft_len=2 passes
```

## Stop Lines

Stop and report if:

- the C128 kernels cannot be built or bound in the current mini environment;
- required state-slot memory cannot be allocated safely;
- accepted commit still drifts after C128 lifecycle is ported;
- exactness only passes with `accepted_kv_copied_tokens == 0`;
- the only exact path recomputes accepted tokens through normal sequential target
  decode, eliminating the target-pass reduction.

If accepted commit still drifts, do not proceed to TARGET 11.3.  Instead report
the first mismatching owner and propose a narrow follow-up bisection target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_online_c128_lifecycle_port/README.md
```

Include:

- source-parity table against SGLang;
- API/kernel changes;
- micro/oracle results;
- memory overhead ledger;
- exactness matrix;
- accepted/verified/rejected token stats;
- committed row category stats, clarifying `accepted_kv_copied_tokens`;
- C128 lifecycle diagnostics;
- whether TARGET 11.3 graph/perf is unblocked.

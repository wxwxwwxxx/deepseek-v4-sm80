# TARGET 11.5: DSV4 SM80 MTP BS4 Accepted-Commit State Parity

## Status

Next after TARGET 11.299.

TARGET 11.299 made real progress on the multi-request target-verify contract:

```text
bs=1 exact
bs=2 exact
mixed active verify lengths no longer crash
row/depth packing no longer looks like the root cause
```

The remaining correctness blocker is smaller and sharper:

```text
bs=4 still diverges after request-local accepted/correction KV commit
```

Do not continue broad row-packing or verify-group-size exploration in this
target.  Treat TARGET 11.299 as sufficient evidence that the next suspect is
accepted-commit state equivalence.

## Goal

Find and fix the first state owner that makes MTP accepted commit non-exact for
the deterministic `bs=4` gate:

```text
TP8 /models/DeepSeek-V4-Flash
page_size=256
draft_len=2
decode_len=8
CUDA graph disabled
PyNCCL disabled
accepted commit enabled
```

The target passes only when:

```text
bs=1/2/4 exact
accepted commit remains enabled
the first mismatching state owner is identified and fixed, or a narrow next
target is written with concrete evidence
```

## Starting Evidence

From:

```text
performance_milestones/target11_mtp_multi_request_verify_contract/README.md
```

Known good:

- row/depth packing matches the observed contract;
- mixed verify lengths are represented by fixed-width padding/masking instead
  of crashing;
- grouping target verify by one request does not remove the `bs=4` drift;
- `bs=1` and `bs=2` exactness can pass with accepted commit enabled.

Known failure:

```text
bs=4 baseline req3 ends: [10177, 4254]
bs=4 MTP req3 ends:      [14486, 361]
```

Important interpretation:

```text
target verify can emit matching tokens inside the verify step, but after
accepted/correction rows are committed, the next normal target decode reads a
different long-lived state.
```

So the likely problem is no longer ordinary attention/GEMM math or target-verify
row order.  The likely problem is one of:

- full KV / SWA tail commit;
- C4 compressed cache or indexer cache commit;
- C128 online MTP pending/write/commit state;
- request metadata lifecycle such as `cached_len`, `device_len`, page-table, or
  component mapping update order;
- correction/bonus row commit count or seq-len off-by-one.

## Key Code References

Mini target verify and commit:

```text
python/minisgl/engine/engine.py
  _make_mtp_flattened_verify_batch
  _forward_mtp_flattened_verify_with_hidden
  accepted commit snapshot/restore/commit block
```

DSV4 attention/cache owners:

```text
python/minisgl/models/deepseek_v4.py
  DSV4 attention forward
  full/SWA KV store
  indexer/compressor store
  C128 MTP prefix write hook

python/minisgl/attention/deepseek_v4.py
  OnlineC128MTPController.prepare_forward
  OnlineC128MTPController.write_prefix_states
  OnlineC128MTPController.commit_pending
  DSV4AttentionBackend.store_compressed
  DSV4AttentionBackend.store_indexer
```

C128 online MTP kernels:

```text
python/minisgl/kernel/deepseek_v4.py
  online_c128_mtp_mark_pending
  online_c128_mtp_write_prefix_states
  online_c128_mtp_commit_pending

python/minisgl/kernel/csrc/jit/dsv4_online_c128_mtp.cu
```

SGLang reference:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

## Work Plan

### 1. Freeze the minimal reproducer

Start by rerunning the exact gate from TARGET 11.299 and record the current
baseline:

```text
bs=1 exact
bs=2 exact
bs=4 not exact, first failing request/token
```

Keep:

- accepted commit enabled;
- graph disabled;
- PyNCCL disabled;
- page size 256;
- draft length 2;
- decode length 8.

Do not hide the bug by:

- disabling accepted commit;
- recomputing accepted rows through sequential normal decode;
- changing sampling behavior;
- only changing target-verify group size;
- skipping correction/bonus rows without proving the SGLang contract.

### 2. Locate the first failing commit event

For `bs=4`, log each target-verify step at request granularity:

- request id / batch index;
- committed length before verify;
- active/padded verify length;
- input rows;
- target rows;
- accepted prefix;
- correction or bonus row;
- `candidate_copy_rows`;
- actual committed locs and positions;
- request lengths before and after `req.complete_one()`;
- next normal target token predicted immediately after commit.

Stop once the first event is found where:

```text
normal decode from baseline prefix != normal decode after MTP committed state
```

The goal is not to collect more macro output.  The goal is to find the first bad
commit.

### 3. Build component parity snapshots

At the first bad commit, compare normal sequential decode state against
target-verify accepted-commit state for the same visible token sequence.

At minimum, compare summaries and selected row tensors for:

- full KV rows / SWA tail rows;
- C4 compressed cache rows;
- C4 indexer cache rows if separate;
- C128 compressed cache rows;
- C128 online MTP state bank 0 and pending banks;
- page table and component page-table rows;
- request scalar state (`cached_len`, `device_len`, emitted tokens).

Use hashes plus max-abs deltas.  For any mismatch, record:

```text
owner
layer id
position
loc
dtype
shape
max_abs_delta
first mismatching element when practical
```

### 4. Isolate state owners with fail-closed toggles

Add temporary diagnostic toggles only as needed.  They should be clearly named
and should not become promoted behavior unless the target proves them.

Useful toggles may include:

```text
MINISGL_DSV4_MTP_COMMIT_DISABLE_C128_ONLINE=1
MINISGL_DSV4_MTP_COMMIT_DISABLE_COMPRESSED=1
MINISGL_DSV4_MTP_COMMIT_DISABLE_INDEXER=1
MINISGL_DSV4_MTP_COMMIT_FORCE_TORCH_C128=1
MINISGL_DSV4_MTP_COMMIT_TRACE_STATE=1
```

The point is attribution, not final performance.  If disabling one owner makes
the next-token oracle exact, that owner becomes the next repair focus.

### 5. Audit C128 online MTP against SGLang

Pay special attention to C128 online state because TARGET 11.299 points at
post-commit state drift and mini has a dedicated C128 online MTP path.

Compare mini vs SGLang for:

- what `pending_seq_lens` stores;
- whether pending state is indexed by request pool index or target-verify row;
- bank layout: committed bank vs pending banks;
- whether accepted length includes correction/bonus rows;
- behavior at chunk boundary and partial chunk;
- state initialization from existing C128 chunk state;
- commit copy source bank selection.

If mini and SGLang differ, prefer porting the SGLang contract unless there is
clear evidence mini's variant is intentional and exact.

### 6. Fix the smallest proven owner

Once the first owner is identified, make the smallest correctness fix.

Examples:

- correct accepted/correction `copy_rows` or seq-len passed to C128 commit;
- copy/update a missing compressed/indexer state row;
- change C128 pending bank source selection;
- update request metadata after all owner commits instead of before;
- restore rejected/padded rows more completely;
- correct component mapping for committed rows.

Do not start CUDA graph or performance work in this target.

### 7. Run a lightweight batch-scaling exposure gate

After `bs=4` is fixed, run a short observation-only gate:

```text
bs=1,2,4,8,16
draft_len=2
decode_len=8
```

Record whether the same owner remains fixed at larger batch sizes.  If a new
failure appears at `bs=8+`, reduce it to the smallest failing batch size and
write the next narrow target.  Do not debug large batch directly unless it is
already the smallest reproducer.

## Success Criteria

Minimum:

```text
bs=1 exact
bs=2 exact
bs=4 exact
accepted commit enabled
first bad commit owner identified
```

Full:

```text
bs=1/2/4 exact
bs=8/16 short exposure has no new correctness failure, or a smallest new
failure is clearly documented
SGLang parity notes explain the fixed owner
TARGET 11.3 graph/perf is either unblocked or still blocked by a named next
correctness target
```

## Stop Lines

Stop and report instead of continuing broad exploration if:

- exactness only passes by disabling accepted commit;
- exactness only passes by sequentially recomputing accepted rows;
- no first bad commit event can be isolated;
- component snapshots show multiple owners diverging before a single first
  owner can be determined;
- `bs=4` passes but `bs=8+` exposes a different failure that requires a new
  minimal reproducer;
- the fix would require a broad SGLang runtime port beyond this target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_bs4_accepted_commit_state_parity/README.md
```

Include:

- exact repro command lines and env flags;
- first bad commit event trace;
- component parity table;
- SGLang C128/commit parity notes;
- fix summary;
- exactness matrix for `bs=1/2/4`;
- batch-scaling exposure summary for `bs=8/16` if the `bs=4` fix lands;
- recommendation for whether TARGET 11.3 can proceed.

# TARGET 11.7: DSV4 SM80 MTP BS6 Path Census And Contract Closure

## Status

Next after TARGET 11.6.

TARGET 11.6 fixed the reduced `bs=5` exposure failure:

```text
bs=1/2/4/5 exact
accepted commit enabled
```

It also proved an important point: the `bs=5 req4` failure was not normal target
batch-shape sensitivity.  It was a target-verify KV-path mismatch:

```text
first owner: layer0.kv_after_kv_norm_rope
cause: single-row target verify was forced onto the separate exact KV store
path instead of matching normal fused q_kv_norm_rope_store semantics.
```

The remaining smallest exposure failure is now `bs=6`:

```text
bs=6 mismatches [0, 3]
req0 first mismatch token index 6
req3 first mismatch token index 3
```

Do not continue as a blind "one batch size at a time" loop.  This target must
first enumerate the source-level and runtime MTP path matrix so we know how many
previously untested branches remain.

## Goal

1. Build a source-derived and runtime-confirmed census of MTP target-verify /
   accepted-commit paths.
2. Use the smallest remaining reproducer, `bs=6`, to close the next correctness
   contract.
3. Decide whether the remaining failures are a small number of fixable path
   branches or evidence that the MTP target-verify runtime needs a broader
   contract unification before graph/perf work.

The target passes only when:

```text
bs=1/2/4/5/6 exact
accepted commit remains enabled
path census identifies covered and still-untested branches
light bs=7/8/16 exposure is rerun and interpreted
TARGET 11.3 go/no-go is updated
```

## Starting Evidence

From:

```text
performance_milestones/target11_mtp_bs5_exposure_state_parity/README.md
```

Fixed:

```text
bs=1 exact
bs=2 exact
bs=4 exact
bs=5 exact
```

Remaining exposure:

```text
bs=6 mismatches:
  req0 first mismatch token index 6
  baseline [11111, 64465, 361, 582, 9628, 3362, 223, 18]
  MTP      [11111, 64465, 361, 582, 9628, 3362, 582, 18]

  req3 first mismatch token index 3
  baseline [24740, 528, 4603, 5071, 11273, 438, 39772, 86103]
  MTP      [24740, 528, 4603, 3605, 19, 438, 39772, 10929]

bs=7 mismatches [0, 3, 4, 6]
bs=8 mismatches [0, 3, 4, 7]
bs=16 mismatches [1, 3, 4, 7, 13]
```

Prefer `bs=6 req3` as the first narrow probe because it diverges earliest
(`token index 3`).  Also keep `bs=6 req0` in the validation table because it may
represent a later accepted-commit or fallback shape.

## Key References

Mini:

```text
python/minisgl/engine/engine.py
  _make_mtp_flattened_verify_batch
  _forward_mtp_flattened_verify_with_hidden
  MTP verify grouping and parent-batch-size tracking
  accepted commit snapshot/restore/commit block

python/minisgl/models/deepseek_v4.py
  DSV4 attention forward
  fused vs separate q/kv norm-rope-store paths
  target-verify flags

python/minisgl/attention/deepseek_v4.py
  DSV4 attention metadata
  target_verify_decode_rows
  store_swa / store_compressed / store_indexer
  OnlineC128MTPController

python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/csrc/jit/dsv4_online_c128_mtp.cu
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/spec_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
```

## Work Plan

### 1. Build the source-level MTP path census first

Before adding another local fix, inspect the source and produce a table of
all MTP target-verify / accepted-commit path axes.

At minimum include:

```text
parent_batch_size:
  <=2, >2, sticky previous parent bs

verify grouping:
  group_size default, explicit MINISGL_DSV4_MTP_VERIFY_GROUP_SIZE,
  sub_batch size, request order

target verify row shape:
  active_verify_len=1/2/3
  padded_verify_len=1/2/3
  single_active_verify_row
  mixed active lengths

attention consumer:
  base target-verify attention
  force_torch_attention
  splitk target_verify_decode_rows

KV store path:
  fused q_kv_norm_rope_store
  separate q_norm_rope + kv_norm_rope_store
  dsv4_force_exact_kv_store
  dsv4_force_torch_attention decoupled from exact store

state owners:
  full/SWA KV
  C4 compressed cache
  C4 indexer cache/state
  C128 compressed cache
  online C128 MTP pending/write/commit
  page table and component loc mapping

request lifecycle:
  cached_len/device_len before verify
  copy_rows / accepted_prefix / correction row / bonus row
  req.complete_one order
  page boundary / C4 boundary / C128 boundary
```

For each row in the table, record:

```text
branch condition
expected exactness contract
covered by bs=1/2/4/5 or not
first target that introduced/fixed it
which runtime traces show it
remaining risk
```

Use `rg` and local source inspection rather than guessing.  The purpose is to
avoid not knowing how many untested paths remain.

### 2. Add a runtime path coverage summary

From existing raw files and a fresh `bs=6` run, summarize which path branches
are actually exercised by:

```text
bs=1/2/4/5 passing gates
bs=6 failing gate
bs=7/8/16 exposure failures
```

Include in the runtime trace:

- parent batch size;
- verify group size;
- active/padded verify length;
- force_torch_attention;
- target_verify_decode_rows;
- dsv4_force_exact_kv_store;
- fused vs separate KV store path if traceable;
- accepted prefix / copy rows;
- first mismatching request/token.

If a source-level branch has no runtime coverage, mark it as untested.

### 3. Freeze the bs=6 reproducer

Use:

```text
TP8 /models/DeepSeek-V4-Flash
page_size=256
num_pages=16
max_running_req=6
decode_len=8
draft_len=2
CUDA graph disabled
PyNCCL disabled
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
accepted commit enabled
```

Record baseline and MTP outputs for `bs=6`.  Keep `bs=1/2/4/5` as regression
controls.

### 4. Probe bs=6 req3 first

For `bs=6 req3`, locate the first event that can explain the token-index-3
divergence.

Use the same triage order that worked in TARGET 11.6:

1. normal target batch-shape oracle for the failing prefix;
2. target-verify vs normal oracle top-k/logits;
3. layer/submodule hidden parity;
4. state parity only if the logits path itself is not the first owner.

Record whether the owner is:

- normal target batch-shape sensitivity;
- target-verify attention consumer;
- KV norm/rope/store path;
- C4/C128/indexer state;
- accepted/correction row commit;
- request lifecycle / metadata.

### 5. Probe bs=6 req0 only after req3 is classified

`req0` diverges later at token index 6.  It may be the same owner as req3 or a
different branch.  Do not spend the whole target on req0 until req3 is
classified.

After the req3 fix or classification, rerun `bs=6`:

- if req0 also becomes exact, mark it covered by the same contract;
- if req0 remains, repeat the same narrow oracle/parity workflow for req0;
- if a new request becomes the first mismatch, reduce to that owner and report.

### 6. Fix or unify the smallest proven contract

If the source/path census shows only one narrow missing branch, fix it locally.

If the census shows several target-verify variants with different numerical
contracts, stop and propose a broader target-verify runtime unification.  A
local patch is acceptable only if it reduces the branch matrix, not if it adds
another special case with unclear coverage.

Do not:

- disable accepted commit;
- sequentially recompute accepted rows;
- make all MTP decoding single-request;
- start CUDA graph/perf work;
- optimize the fast path before exactness is proven.

### 7. Rerun exactness and exposure

After a fix:

```text
bs=1/2/4/5/6 exact
bs=7/8/16 light exposure
```

If `bs=7+` still fails, reduce to the smallest remaining failing batch and
write a narrow next target.  Use the path census to say whether it is a known
untested branch or a new class.

## Success Criteria

Minimum:

```text
source-level path census written
runtime path coverage summary written
bs=1/2/4/5 remains exact
bs=6 either fixed or reduced to a concrete first owner with next target
accepted commit remains enabled
```

Full:

```text
bs=1/2/4/5/6 exact
bs=7/8/16 exposure either passes or fails with a known remaining path branch
TARGET 11.3 remains blocked/unblocked with evidence
```

## Stop Lines

Stop and report if:

- source census shows too many special-case MTP paths to safely patch one more;
- the first bs=6 owner cannot be isolated;
- exactness only passes by disabling accepted commit;
- exactness only passes by sequentially recomputing accepted rows;
- a fix for bs=6 regresses bs=1/2/4/5;
- `bs=6` requires a broad SGLang runtime port rather than a local mini fix.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_bs6_path_census_contract_closure/README.md
```

Include:

- source-level MTP path census table;
- runtime path coverage table;
- exact commands and env flags;
- bs=6 req3 first-owner analysis;
- bs=6 req0 follow-up if needed;
- fix summary or unification recommendation;
- validation matrix for `bs=1/2/4/5/6`;
- exposure matrix for `bs=7/8/16`;
- explicit TARGET 11.3 go/no-go.

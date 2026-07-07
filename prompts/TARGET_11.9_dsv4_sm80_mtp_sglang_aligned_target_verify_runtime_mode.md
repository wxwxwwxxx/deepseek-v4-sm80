# TARGET 11.9: DSV4 SM80 MTP SGLang-Aligned Target-Verify Runtime Mode

## Status

Next after TARGET 11.8.

TARGET 11.8 wrote the target-verify contract and proved that another local
per-batch patch is the wrong direction. No runtime code was promoted there. The
current final code remains the TARGET 11.6 behavior:

```text
bs=1/2/4/5 exact
bs=6 fails for req0 and req3
accepted commit enabled
```

This target should implement, or precisely no-go, a single SGLang-aligned
target-verify runtime mode for DeepSeek V4 MTP on A100/sm80.

## Goal

Introduce one explicit target-verify runtime mode, tentatively:

```text
dsv4_target_verify_runtime = "sglang_prefill_extend"
```

The mode must replace the current scattered numerical selectors for:

- target-verify row width;
- active vs padded row masks;
- attention metadata and backend choice;
- KV producer/store semantics;
- accepted `copy_rows` and rollback/restore ownership;
- online C128 pending/write/commit behavior.

The target passes when:

```text
bs=1/2/4/5/6 exact
accepted commit enabled
target_commit_kv_copies > 0
accepted_kv_copied_tokens > 0
bs=7/8/16 light exposure passes, or fails with a first owner that is not another
batch-size branch
```

Do not start CUDA graph or throughput optimization in this target. This is still
correctness/runtime-contract work.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_runtime_contract_unification/README.md
performance_milestones/target11_mtp_bs6_path_census_contract_closure/README.md
performance_milestones/target11_mtp_bs5_exposure_state_parity/README.md
performance_milestones/target11_mtp_bs4_accepted_commit_state_parity/README.md
```

TARGET 11.8 final conclusion:

```text
current 11.6:
  parent>2 active 2/3 -> force_torch attention + separate exact KV store
  passes bs=4/5, fails bs=6 req0/req3

KV-unify experiment:
  parent>2 active 2/3 -> force_torch attention + fused KV store
  fixes bs=6 req3, regresses bs=4/5, leaves bs=6 req0

attention-unify experiment:
  parent>2 active 2/3 -> split-k target_verify attention + fused KV store
  fixes bs=6 req0, regresses bs=6 req3/req4
```

Interpretation:

```text
parent batch size, active verify length, attention backend, and KV producer are
currently selecting numerical semantics. They must become scheduling/perf
choices only after their implementations are parity-tested.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
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
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
```

Prefer SGLang's DeepSeek V4 MTP target-verify behavior when mini and SGLang
differ, unless mini has a written reason to intentionally diverge.

## Non-Goals

- Do not optimize CUDA graph replay.
- Do not tune throughput.
- Do not add a new per-batch or observed-token special case.
- Do not disable accepted commit to make text exact.
- Do not sequentially recompute accepted rows after verification and call that
  the final MTP runtime.
- Do not make `bs=6` pass by regressing `bs=1/2/4/5`.
- Do not make parent batch size, active verify length, or request slot choose
  different numerical semantics.

## Required Runtime Contract

### 1. Central Contract Object

Create a single owner for DSV4 MTP target-verify runtime semantics. It can be an
enum, dataclass, config field, or a small internal mode object, but it must
centralize the current scattered decisions.

Suggested name:

```text
dsv4_target_verify_runtime = "sglang_prefill_extend"
```

It should own at least:

- target verify width;
- active row mask;
- padded row mask;
- row-to-request mapping;
- row depth and position mapping;
- attention metadata mode;
- target KV producer mode;
- component/C4/C128 store policy;
- accepted commit `copy_rows`;
- online C128 pending/write/commit decision.

Existing debug flags may remain as diagnostics, but the production candidate
must not depend on them as independent numerical selectors:

```text
dsv4_force_exact_kv_store
dsv4_force_torch_attention
dsv4_target_verify_decode_rows
_dsv4_mtp_max_parent_batch_size
MINISGL_DSV4_MTP_VERIFY_SPLITK_ATTENTION
MINISGL_DSV4_MTP_VERIFY_FORCE_TORCH_ATTENTION
```

### 2. SGLang-Aligned Verify Metadata

Build target-verify metadata like a prefill/extend group rather than a decode
path selected by parent batch size.

The expected shape is:

```text
seq_lens += verify_width
extend_lens = [verify_width] * verify_batch
request indices are repeated per verify row
row d has causal length committed_seq_len + d + 1
```

For top-k 1 and `draft_len=2`, each request has up to three verify rows:

```text
row 0 input = first target token / bonus seed from the current target decode
row 1 input = draft_0
row 2 input = draft_1
```

Active rows and padded rows must be separate concepts:

- active rows can affect output, accepted commit, component store, and online
  C128 commit;
- padded rows exist only for scheduling/shape and must be harmless;
- no attention row may read another request's rows or a padded/future row.

### 3. Unified Attention Consumer

The target-verify attention result must match normal target decode on the same
visible prefix.

The backend name is not the contract. `force_torch`, split-k target verify, base
sparse, or a future SGLang-style backend are allowed only after they are shown to
implement the same metadata semantics.

For this target, prioritize correctness over speed. A slower reference checker
is acceptable, but the candidate runtime must still be a real target-verify path
with accepted commit enabled.

Add a debug checker if useful:

```text
chosen target-verify attention output
vs torch/reference attention output
for selected rows/layers
```

### 4. Unified KV Producer And Store

Use one target DSV4 KV producer semantic for all active target-verify rows, all
active verify lengths, and all parent batch sizes.

SGLang parity points to the normal target DSV4 path as the owner:

```text
normal decode uses fused q/k norm + RoPE + store
target verify uses the same producer for active 1/2/3 rows
```

If a fallback separate norm/RoPE/store path is needed as a correctness oracle,
it must be forced consistently for all target-verify shapes. It must not be
selected by:

- parent batch size;
- active verify length;
- verify group size;
- request slot;
- observed token value.

Accepted target-verify rows must produce long-lived state equivalent to normal
target decode for:

- full/SWA KV;
- C4 compressed cache;
- C4 indexer cache/state;
- C128 compressed cache;
- online C128 MTP pending/write/commit;
- page/component mapping;
- request `cached_len` and `device_len`.

### 5. Accepted Commit

Keep mini's snapshot/rollback/restore model if it remains the smallest safe
implementation:

1. Snapshot pre-verify state.
2. Run target verify into verify/temp locations.
3. Determine `accepted_prefix`, emitted tail, and `copy_rows`.
4. Snapshot committed rows.
5. Roll back all verify rows.
6. Restore only committed rows.
7. Commit online C128 pending state for accepted sequence lengths.
8. Call `req.complete_one()` only after committed state is restored.

Move all `copy_rows` decisions behind the runtime contract and assert that every
copied row is active and visible.

Do not silently fall back to non-committed MTP. If a page/C4/C128 boundary is
unsupported, fail closed and document the boundary.

## Work Plan

### 1. Re-read TARGET 11.8 as the contract source

Start from the written contract in:

```text
performance_milestones/target11_mtp_target_verify_runtime_contract_unification/README.md
```

Create a short design note in the new milestone report that states:

- which 11.8 contract clauses are implemented directly;
- which clauses are still diagnostic-only;
- which clauses need a follow-up target.

### 2. Map SGLang source behavior to mini code

Produce a source-parity table:

```text
concept
SGLang behavior
mini before 11.9
mini after 11.9
same/different
correctness risk
```

Focus only on DeepSeek V4 MTP target verify:

- frozen-KV draft state;
- target verify metadata;
- target verify attention;
- KV/component/C4/C128 producer/store;
- accepted row movement and commit;
- online C128 MTP lifecycle.

### 3. Implement the unified mode

Make the smallest code change that removes the current numerical branch split.

Expected areas:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
```

Implementation guidance:

- build explicit row/depth/active/padded metadata once;
- pass that metadata through the model/attention path rather than re-deriving
  numerical behavior from batch shape;
- make padded rows harmless and excluded from commit;
- use the normal target KV producer semantic for every active verify row;
- leave debug envs available for attribution, but keep the candidate runtime
  selected by one mode.

### 4. Validate exactness first

Run eager/no-graph tests with:

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
```

Required matrix:

```text
bs=1 exact
bs=2 exact
bs=4 exact
bs=5 exact
bs=6 exact
```

Required commit stats:

```text
accepted_kv_commit_fail_closed = false
target_commit_kv_copies > 0
accepted_kv_copied_tokens > 0
```

Then run light exposure:

```text
bs=7
bs=8
bs=16
```

If exposure fails, identify the first failing request/token and first owner. It
is acceptable to defer a newly discovered boundary, but not another
batch-size-selected numerical contract.

### 5. Add one boundary-focused gate if feasible

Add or run at least one short test that is likely to cross one of these:

- page boundary;
- C4 boundary;
- C128 boundary;
- mixed active verify lengths in the same parent batch.

If a real C128 boundary test is too slow for this target, document the missing
gate as a blocker before TARGET 11.3.

### 6. Decide whether TARGET 11.3 can start

TARGET 11.3 may start only if:

```text
bs=1/2/4/5/6 exact
accepted commit enabled
bs=7/8/16 exposure is clean or has a non-contract follow-up
one boundary gate is clean or explicitly fail-closed
```

Otherwise, write the next correctness target with the first unresolved owner.

## Success Criteria

Minimum:

```text
one explicit target-verify runtime mode exists
parent batch size is not a numerical selector
active verify length is not a KV producer selector
bs=1/2/4/5 remains exact
accepted commit remains enabled
```

Full:

```text
bs=1/2/4/5/6 exact
bs=7/8/16 exposure passes or finds a new concrete non-batch-branch owner
no new per-batch special case is introduced
SGLang parity table is updated with mini-after-11.9 behavior
TARGET 11.3 go/no-go is clear
```

## Stop Lines

Stop and report if:

- the only passing fix is another `if bs == ...` or parent-size branch;
- exactness requires disabling accepted commit;
- exactness requires sequentially recomputing accepted rows as the final path;
- `bs=1/2/4/5` regresses;
- the unified mode cannot be written without a broader SGLang runtime port;
- C4/C128/component ownership becomes ambiguous or silently non-committed;
- a debug fallback becomes the only passing runtime and cannot be explained as
  SGLang-equivalent.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_sglang_aligned_target_verify_runtime_mode/README.md
```

Include:

- design note for the unified runtime mode;
- SGLang parity table with mini-before and mini-after columns;
- implementation summary;
- removed or centralized branch list;
- exactness matrix for `bs=1/2/4/5/6`;
- light exposure matrix for `bs=7/8/16`;
- commit stats proving accepted commit remains enabled;
- boundary gate result or explicit follow-up;
- TARGET 11.3 go/no-go recommendation.

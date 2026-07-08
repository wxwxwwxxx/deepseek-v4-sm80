# TARGET 11.249: DSV4 SM80 MTP Online C128 Main-State Contract Port

## Status

Next after TARGET 11.248.

TARGET 11.248 classified the remaining MTP exactness owner as:

```text
primary: c128_disabled_contract_owner
symptom: c128_write_skipped_owner
```

The first failing anchor is:

```text
MTP trace_index=113
event=mtp_after_normal_before_verify
uid=0
position=3
cached_len=5

baseline trace_index=96
event=baseline_after_normal_decode

component: c128_attention_state.layer3
component loc: 1539
mapping: full/SWA/C4/C128/page-table aligned
baseline: nonzero, sha=4fde0338954bee2b, abs_sum=0.21875
MTP:      zero,    sha=e5a00aa9991ac8a5, abs_sum=0.0
```

11.248 also found that Mini keeps online C128 MTP banks in a separate side
buffer, while SGLang stores committed and pending online C128 banks in the main
C128 compress-state pool.  The current Mini layout gives target verify an
internally active side state, but does not publish the same no-spec-equivalent
state surface that normal decode, state parity, and later attention reads
reason about.

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes.

## Goal

Port or redesign Mini's online C128 MTP state lifecycle so it obeys a single
main-state contract:

```text
For C128 layers, the state used by normal decode, target verify prefix writes,
accepted commit, snapshot/restore, state parity, and later attention reads must
be one coherent C128 component-state surface.
```

Prefer an SGLang-aligned implementation.  If exact source parity is not possible
inside Mini's current SM80 backend, implement a fail-closed variant that makes
C128 state consumption explicit and proves greedy exactness before any
performance work.

The target passes when one of these outcomes is reached:

1. `main_state_contract_fixed`: Mini's C128 online MTP state is published through
   the main C128 component-state contract, focused anchors no longer show the
   loc1539 zero/nonzero mismatch, and greedy exactness passes the required
   matrix.
2. `fail_closed_exact`: C128 online MTP is safely disabled or bypassed under MTP
   with downstream metadata/read behavior also aligned, and greedy exactness
   passes.  This is acceptable as a short-term correctness step but must report
   expected performance cost.
3. `layout_port_no_go`: source parity shows a larger layout/kernel contract
   change is required than can be safely completed in this target.  The report
   must include a precise implementation plan and the smallest next target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_c128_component_state_publication_parity/README.md
performance_milestones/target11_mtp_c128_component_state_publication_parity/raw/
performance_milestones/target11_mtp_accepted_commit_state_parity_after_moe_microbatch/README.md
performance_milestones/target11_mtp_target_verify_moe_normal_shape_microbatch_runtime/README.md
prompts/TARGET_11.248_dsv4_sm80_mtp_c128_component_state_publication_parity.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not patch accepted commit mapping; 11.247 ruled it out as first owner.
Do not patch logits/sampler.
Do not start graph/perf, CUDA graph capture, PyNCCL, communication-policy work,
or low-precision research.
Do not branch on uid0, pos3, layer3, loc1539, bs6, request id, token, rank, or
prompt text.
```

## References

Mini:

```text
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
```

Mini paths to audit and likely edit:

```text
DSV4CompressStatePool
DSV4KVCache.get_online_c128_mtp_state*
OnlineC128MTPController.prepare_forward
OnlineC128MTPController.write_prefix_states
OnlineC128MTPController.commit_pending
DSV4AttentionBackend.write_c128_mtp_prefix_states
compressor/store_compressed C128 paths
_snapshot_mtp_kv_rows
_mtp_online_c128_state_summary
state parity trace hooks
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
```

Use SGLang as the reference for behavior, not merely field names.  In
particular, confirm:

```text
where the main C128 kv_score_buffer is allocated;
whether online C128 changes row count, last dimension, or both;
which slice is normal committed state and which slices are pending/draft banks;
when prepare_forward commits pending state;
where write_prefix_states publishes target-verify prefix banks;
how frozen-KV target verify avoids corrupting committed target state;
how target verify rollback/replay interacts with C128 state.
```

## Non-Goals

- Do not tune performance.
- Do not enable CUDA graph for MTP.
- Do not change sampling behavior.
- Do not implement INT8/FP8 research.
- Do not change PyNCCL or communication routing.
- Do not make a local copy from the observed baseline row into MTP state as the
  fix.  11.248 noted that the baseline row may include residency or
  initialization ambiguity; the fix must be a state lifecycle contract.

## Work Plan

### 1. Source-Parity Contract

Write down the exact C128 online MTP state contract from SGLang:

```text
main C128 pool shape and dtype
bank layout: committed state, pending/draft state, extra state dimensions
commit order before non-idle forward
write_prefix_states source/destination
target-verify temporary state ownership
snapshot/restore or rollback semantics
downstream attention read surface
```

Then compare Mini's current implementation:

```text
main C128 kv_score_buffer shape and dtype
online_mtp_state side-buffer shape and dtype
controller ready/prepare/write/commit behavior
normal decode behavior without dsv4_target_verify_metadata
state trace surface
downstream C128 attention read surface
```

The report must explicitly state whether Mini can port the SGLang layout
directly, or whether a compatibility layer is needed because kernels currently
assume a `2 * head_dim` C128 main row.

### 2. Choose The Correctness Strategy

Prefer Strategy A.

Strategy A: SGLang-aligned main-state port.

```text
Make online C128 committed/pending state live in, or publish through, the main
C128 component-state pool.
Ensure commit_pending publishes to the state surface later attention reads.
Ensure target-verify prefix writes and normal target decode use compatible state
locations.
Make snapshot/restore inspect the same state surface.
Keep side-buffer only if it is a private staging buffer with an explicit,
tested publish step into main state before any read/snapshot that requires it.
```

Strategy B: Fail-closed exactness.

```text
Disable or bypass online C128 under MTP only if downstream metadata/read paths
also stop consuming stale or uninitialized C128 state, and the full greedy
matrix becomes exact.
Document expected performance cost and leave a follow-up port target.
```

Do not mix the strategies silently.  The milestone must say which strategy won
and why.

### 3. Implement Minimal Contract Fix

Allowed changes:

```text
pool layout/allocation for C128 online MTP;
online C128 controller state binding;
prefix-state write destination;
pending commit destination and order;
snapshot/restore surfaces for C128 online state;
debug trace hooks that verify the contract;
guarded fallback or fail-closed path if source parity cannot be completed.
```

Required implementation checks:

```text
No uid/position/layer/loc/batch/token special cases.
No copying from baseline trace artifacts.
No random/uninitialized C128 state consumption.
No stale side-buffer state surviving request cleanup.
No shape assumptions that break C4, SWA, C128 boundary storage, or prefix cache.
No graph-capture-only behavior changes.
```

If changing the main C128 row shape, audit all users that slice or reshape the
C128 pool.  Preserve existing non-MTP C128 behavior and keep text sanity for the
non-MTP baseline.

### 4. Focused Anchors

Re-run the TARGET 11.248 anchor after the fix:

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
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1
```

Required anchor checks:

```text
trace96/trace113-equivalent C128 anchor no longer has aligned-loc zero/nonzero
  mismatch at the first failing component;
main C128 state and online C128 state summary refer to the same published
  lifecycle or have an explicit staging/publish explanation;
bs1/2/4/5 remain exact;
bs6 improves or the first owner moves to a different component with evidence.
```

If the first owner moves, follow it only far enough to prove this target fixed
the C128 contract.  Do not start a new deep owner chase inside this target.

### 5. Required Validation

Correctness:

```text
focused C128 lifecycle trace from 11.248 anchor
bs1/2/4/5/6 greedy exactness matrix
text sanity smoke with MTP enabled
non-MTP baseline text sanity smoke
```

Regression guards:

```text
focused bs2 MoE microbatch guard from TARGET 11.246
accepted-commit state/KV parity guard from TARGET 11.247
state trace check that C128 loc mapping remains aligned
request cleanup check that C128 online state does not leak across requests
```

Static checks:

```bash
python -m py_compile \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/utils/dsv4_mtp_debug.py

git diff --check
```

Optional, only after exactness passes:

```text
rough memory ledger for any main-state pool growth;
rough runtime note if fail-closed C128 disables an optimization.
```

## Stop Conditions

Stop and write the milestone report when any of these happens:

1. The C128 main-state contract is fixed and the full exactness matrix passes.
2. The C128 main-state contract is fixed, the old C128 anchor is gone, and a new
   first owner appears outside this target's scope.
3. A fail-closed exact path passes, with a clear follow-up for the SGLang-aligned
   main-state port.
4. Source parity proves Mini needs a larger kernel/layout port.  In this case,
   do not keep patching around the issue; write the precise next target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
```

The report must include:

- final classification: `main_state_contract_fixed`, `fail_closed_exact`, or
  `layout_port_no_go`;
- SGLang-vs-Mini C128 contract table;
- implementation summary and changed files;
- before/after anchor evidence for `c128_attention_state.layer3`;
- exactness matrix and smoke results;
- memory/performance note if layout changed or C128 was fail-closed;
- remaining risks and the next target recommendation.

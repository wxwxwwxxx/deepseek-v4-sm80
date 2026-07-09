# TARGET 11.248: DSV4 SM80 MTP C128 Component-State Publication Parity

## Status

Next after TARGET 11.247.

TARGET 11.247 found the first remaining bs6 owner after the MoE microbatch fix:

```text
classification: component_state_owner
component: c128_attention_state.layer3
```

The earliest divergence is before target-verify event0 commits anything:

```text
MTP trace_index=113
event=mtp_after_normal_before_verify
uid=0
position=3
cached_len=5

baseline trace_index=96
event=baseline_after_normal_decode
```

Mapping is aligned:

```text
full loc: aligned
SWA loc: aligned
C4 state loc: aligned
C128 state loc: aligned
page table window: aligned
```

But the C128 value differs:

```text
c128_attention_state.layer3
component loc: 1539
baseline: nonzero, sha=4fde0338954bee2b, abs_sum=0.21875
MTP:      zero,    sha=e5a00aa9991ac8a5, abs_sum=0.0
```

This rules out accepted-commit row mapping as the first owner.  The likely
failure is C128 component-state publication/lifecycle under MTP normal target
decode before the first target-verify event.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find why no-spec normal decode publishes nonzero C128 attention state at
`uid0/pos3/layer3/loc1539`, while MTP normal-before-verify leaves the aligned
C128 state zero.

The target should answer:

```text
Is C128 missing because MTP normal decode never writes it, writes it to the
wrong place, writes it and then restores/clears it, or intentionally disables
online C128 in a way that is incompatible with greedy exactness?
```

The target passes when it produces one of these classifications:

1. `c128_write_skipped_owner`: MTP normal decode does not call/write the C128
   component state for rows where no-spec baseline does.
2. `c128_wrong_loc_owner`: MTP writes C128 state, but to a different component
   loc than the aligned baseline loc.
3. `c128_value_owner`: MTP writes to the correct loc, but the computed C128
   state value differs before publication.
4. `c128_restore_clear_owner`: MTP writes the correct value, then snapshot
   restore, rollback, pending-state cleanup, or another lifecycle step clears
   it before the later read.
5. `c128_disabled_contract_owner`: Mini intentionally keeps online C128 disabled
   under MTP, but downstream attention still expects no-spec-equivalent C128
   state.
6. `c128_metadata_owner`: C128 state values are correct, but c128 loc metadata,
   lengths, page table, or component ownership metadata diverges.
7. `instrumentation_no_go`: current hooks cannot observe the C128 write,
   publish, restore, and read lifecycle for the aligned row.

If a minimal source-aligned fix is clear, it may be attempted.  Otherwise close
with the owner and a narrower repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_accepted_commit_state_parity_after_moe_microbatch/README.md
performance_milestones/target11_mtp_accepted_commit_state_parity_after_moe_microbatch/raw/
performance_milestones/target11_mtp_target_verify_moe_normal_shape_microbatch_runtime/README.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not patch accepted commit mapping; 11.247 found the first divergence before
event0 target-verify commit.
Do not start graph/perf, CUDA graph capture, low-precision research, PyNCCL, or
communication-policy work.
Do not branch on uid0, pos3, layer3, loc1539, bs6, request id, token, rank, or
prompt text.
```

## References

Mini:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/kernel/deepseek_v4.py
```

Likely C128 paths:

```text
OnlineC128MTPController
prepare_forward
mark_pending
write_prefix_states
commit_pending
store_compressed
c128_cache / c128_page_indices / c128_topk_lengths
online_c128_mtp pending seq lens and state slots
state parity snapshot
target-verify temporary writes and restores
normal target decode before target verify
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/model_executor/pool_configurator.py
```

SGLang warning to verify:

```text
Online C128 assumes a strict forward-only schedule.  Speculative decode may
require rollback/replay or disabling online C128 unless the runtime preserves
no-spec-equivalent component state.
```

## Non-Goals

- Do not change MoE microbatch behavior except to fix a proven regression.
- Do not patch logits/sampler.
- Do not disable accepted commit to pass exactness.
- Do not start graph/perf or CUDA graph work.
- Do not special-case `uid0/pos3/layer3/loc1539`.
- Do not rely on isolated bs6; the failure is from the full `1/2/4/5/6`
  schedule.

## Work Plan

### 1. Reproduce First C128 Divergence

Use TARGET 11.247's runtime state:

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
bs1/2/4/5 exact
bs6 req4/req5 fail
first state divergence: uid0 pos3 layer3 c128 loc1539 baseline nonzero vs MTP zero
mapping aligned for full/SWA/C4/C128/page table
```

If the first divergence moves, follow the new first C128 divergence and explain
why the old anchor is stale.

### 2. C128 Publication Timeline

For the aligned anchor and neighboring controls, capture the C128 lifecycle in
both no-spec baseline and MTP:

```text
before layer3 compressor / attention state production
after compressor output for the row
before store_compressed / write_prefix_states
after store_compressed
after online_c128_mtp.write_prefix_states
after online_c128_mtp.mark_pending
after online_c128_mtp.commit_pending
after pre-verify snapshot restore, if applicable
after committed restore, if applicable
before later attention read
```

Track:

```text
uid
position
layer
full loc
c128 component loc
c128 page index
c128 length
state checksum / abs_sum / zero flag
whether write/pending/commit path was invoked
```

Use neighboring rows as controls:

```text
uid0 pos3 loc1539 first bad row
uid0 pos4 loc1540 same request tail
uid4/uid5 later C128 rows from 11.247
one row where no C128 is legitimately absent, if available
```

### 3. Write-Skipped Versus Wrong-Loc Split

Classify the first divergence:

```text
write skipped:
    no C128 write/publish hook fires in MTP for an aligned row where baseline
    writes nonzero state.

wrong loc:
    MTP writes nonzero state, but not to the aligned c128 loc.

wrong value:
    MTP writes to the correct loc, but value differs from baseline before any
    restore/cleanup.

restore/clear:
    MTP writes correct value, then a later lifecycle step zeroes or replaces it.
```

If MTP intentionally disables online C128 in this mode, determine whether
attention later still consumes C128 metadata/cache as if it were present.

### 4. Normal Decode Versus Target Verify Scope

Because the first divergence occurs at `mtp_after_normal_before_verify`, split
normal target decode from target-verify lifecycle:

```text
MTP normal target decode before event0
MTP draft forward, if it can affect shared component pools
MTP target-verify temporary writes, after event0 and later
accepted commit restore
```

The first fix should target normal-before-verify C128 publication unless new
evidence shows draft/verify mutated the shared state before the snapshot.

### 5. Source-Parity / Contract Decision

Write a source-parity table:

```text
Question
SGLang behavior
Mini current behavior
Decision
```

Answer:

```text
Does SGLang enable online C128 under speculative decode for this model/path?
If disabled, does SGLang also change attention metadata so C128 is not consumed?
If enabled, how does SGLang roll back/replay C128 state across draft/verify?
What state belongs to target-owned pools versus draft-owned views?
When are pending C128 writes committed relative to req_to_token/seq_len update?
```

### 6. Minimal Fix Policy

Allowed fixes:

```text
publish normal-target C128 state under MTP using the same path as no-spec
restore/replay C128 state around target verify so committed state matches baseline
disable C128 consumption under MTP only if metadata/read path is also made
source-aligned and exact
fix C128 loc translation or pending commit order if proven wrong
```

Forbidden fixes:

```text
special-case uid0/pos3/layer3/loc1539
branch on bs6, req4, req5, event13, event15, token, rank, or prompt text
copy no-spec baseline state into MTP debug path
disable accepted commit
turn off MoE microbatch to hide the issue
patch final sampled token
```

### 7. Validation

After attribution, and after any minimal fix if attempted:

```text
first C128 divergence anchor no longer baseline nonzero vs MTP zero
focused bs2 MoE microbatch guard remains exact
full bs1/2/4/5/6 matrix with MoE microbatch enabled
bs6 event/state bisection from TARGET 11.247 or equivalent
event13/event15 oracle guard
accepted commit stats
```

If C128 is disabled or bypassed as the chosen fix, include a correctness and
performance risk note.

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

## Deliverables

Write:

```text
performance_milestones/target11_mtp_c128_component_state_publication_parity/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
C128 publication timeline
write-skipped / wrong-loc / wrong-value / restore-clear split
normal decode versus target-verify scope
SGLang source-parity / contract decision
first owner classification
validation results
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The first C128 divergence cannot be reproduced with aligned loc metadata.
- The C128 write/publish lifecycle cannot be instrumented enough to classify
  write-skipped, wrong-loc, wrong-value, or restore-clear.
- SGLang's contract shows a larger speculative C128 rollback/replay port is
  required; close with that port target rather than patching locally.
- A proposed fix only works by branching on uid, position, layer, loc, batch
  size, event, request id, token, rank, or prompt text.
- A safe fix improves bs6 but regresses bs1/2/4/5, focused MoE microbatch
  exactness, or accepted commit accounting.

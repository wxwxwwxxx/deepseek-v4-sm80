# TARGET 11.250: DSV4 SM80 MTP Online C128 Read-Surface Port

## Status

Next after TARGET 11.249.

TARGET 11.249 reached:

```text
classification: fail_closed_exact
blocker: c128_online_main_state_compressor_read_surface_not_ported
```

Important progress from 11.249:

```text
Mini now allocates online C128 MTP banks on the main C128 kv_score_buffer.
get_online_c128_mtp_state() returns the main buffer.
normal decode can publish committed C128 state to bank 0.
pending banks are addressed by state_slot_offset.
clear_state_locs() clears bank 0 and pending banks.
bs1/2/4/5/6 greedy matrix is exact only because target verify and accepted
commit fail closed.
```

Remaining blocker:

```text
Mini does not yet have SGLang's online C128 compressor/decode planner/read
surface.  Accepted commit is intentionally disabled before target verify, so
MTP proposes drafts but verifies and accepts none.
```

TARGET 11.3 graph/perf promotion remains no-go until greedy exactness passes
with target verify and accepted commit active.

## Goal

Port the SGLang-aligned online C128 read/planner surface into Mini so that
C128 attention/compressor code consumes the main online C128 state layout
introduced by TARGET 11.249.

The target should make this true:

```text
normal decode, target verify, accepted commit, snapshot/restore, state parity,
and later attention reads all use the same online C128 main-state contract.
```

Then remove the fail-closed blocker only when the read surface is coherent and
the greedy exactness gates pass.

The target passes with one of these outcomes:

1. `online_c128_read_surface_fixed`: fail-closed blocker removed, target verify
   and accepted commit are active, C128 main-state read/write lifecycle is
   coherent, and bs1/2/4/5/6 greedy exactness passes.
2. `online_c128_read_surface_partial_owner`: old fail-closed blocker is removed
   or narrowed, but a new first owner appears.  The report must prove C128 read
   surface is no longer the first blocker and write the smallest next target.
3. `online_c128_read_surface_no_go`: source parity or kernel/layout audit shows
   a larger kernel/planner port is required.  The report must identify the
   missing primitive and produce a precise follow-up.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_online_c128_main_state_contract_port/README.md
performance_milestones/target11_mtp_online_c128_main_state_contract_port/raw/
performance_milestones/target11_mtp_c128_component_state_publication_parity/README.md
prompts/TARGET_11.249_dsv4_sm80_mtp_online_c128_main_state_contract_port.md
prompts/TARGET_11.248_dsv4_sm80_mtp_c128_component_state_publication_parity.md
prompts/TARGET_11.247_dsv4_sm80_mtp_accepted_commit_state_parity_after_moe_microbatch.md
prompts/TARGET_11.246_dsv4_sm80_mtp_target_verify_moe_normal_shape_microbatch_runtime.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Enable MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH=1 unless explicitly
running a before/after comparison.
Preserve TARGET 11.249's main-state C128 storage contract unless source parity
proves it is wrong.
Do not reopen MoE row-shape work unless focused guards prove it regressed.
Do not patch logits/sampler.
Do not start CUDA graph/perf, PyNCCL, communication-policy work, or low
precision research.
Do not branch on uid, position, layer, loc, bs, request id, token, rank, or
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
tests/core/test_deepseek_v4_kvcache.py
tests/kernel/test_deepseek_v4_wrappers.py
```

Mini paths to audit and likely edit:

```text
_mtp_accepted_commit_blocker
_fail_closed_mtp_exact
_verify_mtp_spec_drafts_flattened
OnlineC128MTPController.prepare_forward/write_prefix_states/commit_pending
OnlineC128MTPController.write_committed_states
DSV4AttentionBackend.write_c128_mtp_prefix_states
C128 compressor decode/store paths
C128 attention read path
state trace and snapshot/restore hooks
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor_v2.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
```

Source behaviors to identify before patching:

```text
CompressorDecodePlan.generate_online or equivalent planner behavior;
_use_online_compress(128) or equivalent enable/disable condition;
which metadata tells C128 attention to read online state rather than legacy
  boundary-only state;
how online C128 handles non-boundary tokens inside a 128-token block;
how target verify reads frozen committed state while writing temporary pending
  state;
where pending banks become bank 0 after accepted commit;
which kernels expect 2 * head_dim rows and which expect 3 * head_dim rows.
```

## Non-Goals

- Do not tune throughput.
- Do not enable CUDA graph for MTP.
- Do not change sampling behavior.
- Do not implement INT8/FP8 research.
- Do not change PyNCCL or communication routing.
- Do not revert TARGET 11.249's main-state storage unless source parity proves
  the storage contract itself is wrong.
- Do not pass exactness by leaving target verify permanently fail-closed.

## Work Plan

### 1. Read-Surface Contract Census

Build a Mini-vs-SGLang table for the C128 online read path:

```text
planner input metadata;
state loc calculation;
bank selection for committed versus pending rows;
read tensor shape and dtype;
non-boundary token behavior;
target-verify frozen read behavior;
normal decode update behavior;
accepted commit publish behavior;
snapshot/restore surface;
cleanup/clear behavior.
```

The table must explicitly answer:

```text
Does Mini attention currently read bank 0 from the online main C128 state?
Does Mini target verify read committed bank 0 while writing pending banks?
Does Mini's compressor/store path still assume legacy 2 * head_dim rows?
Which exact code path requires the blocker in _mtp_accepted_commit_blocker?
```

### 2. Minimal Read-Surface Port

Implement the smallest SGLang-aligned path that makes the blocker unnecessary.

Preferred direction:

```text
Add or adapt a Mini online C128 decode planner that maps full_locs to chunk
state_locs and bank ids.
Make C128 attention/compressor decode consume bank 0 for committed state.
Make target verify consume committed state read-only while write_prefix_states
writes pending banks.
Make commit_pending publish pending banks into bank 0.
Keep snapshot/restore and state trace on the same main-state surface.
```

Allowed changes:

```text
metadata fields for C128 online decode/read;
planner functions for chunk state locs and bank ids;
attention/compressor read selection;
kernel wrapper shape checks for 3 * head_dim state rows;
engine blocker logic and exactness gates;
state trace instrumentation needed to prove the lifecycle.
```

Required constraints:

```text
No uid/position/layer/loc/batch/token special cases.
No copying from baseline trace artifacts.
No random or uninitialized C128 state consumption.
No stale pending bank reads after rejected drafts.
No accepted commit before pending banks have been written and validated.
No regression of non-MTP baseline C128 behavior.
```

### 3. Remove Or Narrow The Blocker

Only remove:

```text
c128_online_main_state_compressor_read_surface_not_ported
```

after a focused lifecycle trace proves:

```text
normal decode writes bank 0;
target verify reads bank 0 as committed/frozen state;
target verify writes pending banks;
accepted commit copies only accepted pending banks into bank 0;
rejected pending banks do not become visible;
later attention reads the published bank 0 state.
```

If only part of the path is ported, replace the blocker with a narrower reason
instead of silently enabling accepted commit.

### 4. Focused Validation

Use the same baseline shape as 11.249:

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

Required checks:

```text
fail_closed_exact_batches == 0 after enabling the port;
target_verify_calls > 0;
draft_tokens_verified > 0;
accepted commit path is either active or explicitly no-go with a narrower
  blocker;
C128 lifecycle trace shows main-state read/write/commit coherency;
bs1/2/4/5/6 greedy exactness matrix passes, or the first new owner is reported
with state evidence.
```

### 5. Regression Guards

Run:

```text
MTP text sanity;
non-MTP baseline text sanity;
focused 11.246 MoE microbatch guard once target verify is active again;
11.247 accepted-commit state/KV guard;
C128 online lifecycle unit tests;
request cleanup check for pending banks and bank 0.
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

Optional only after exactness:

```text
rough memory ledger for online C128 main state;
rough target-pass acceptance stats;
do not run CUDA graph/perf promotion yet.
```

## Stop Conditions

Stop and write the milestone report when any of these happens:

1. The C128 read surface is ported, fail-closed is removed, and bs1/2/4/5/6
   greedy exactness passes.
2. The C128 read surface is ported enough to remove the old blocker, but a new
   first owner appears.  Do not chase it deeply; classify it and propose the
   next target.
3. Source parity proves a missing kernel/planner primitive is required.  Do not
   keep adding speculative local patches; write the precise primitive and next
   implementation target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_online_c128_read_surface_port/README.md
```

The report must include:

- final classification: `online_c128_read_surface_fixed`,
  `online_c128_read_surface_partial_owner`, or `online_c128_read_surface_no_go`;
- SGLang-vs-Mini read/planner surface table;
- implementation summary and changed files;
- before/after blocker status and MTP stats;
- C128 lifecycle trace evidence;
- bs1/2/4/5/6 exactness matrix;
- text sanity and regression guard results;
- remaining risks and next target recommendation.

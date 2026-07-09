# TARGET 11.246: DSV4 SM80 MTP Target-Verify MoE Normal-Shape Microbatch Runtime

## Status

Next after TARGET 11.245.

TARGET 11.245 proved that the only tested source-plausible MoE execution shape
that makes the current target-verify anchors exact under Mini's active backend
is a normal-shape-compatible microbatch contract:

```text
classification:
  normal_shape_compatible_microbatch_contract /
  expert_backend_reduce_cast_row_shape_owner

baseline normal writer shape: [2, 4096]
MTP target-verify shape:      [6, 4096]
winning oracle:               3 x [2, 4096], reassembled to [6, 4096]
MoE backend:                  fused runner / marlin_wna16
```

Oracle exactness:

```text
target actual current:          loc263/264/266 drift, loc267 exact
target full [6,*]:              loc263/264 drift, loc266 exact, loc267 broken
target active-only:             same as full; no padding in this repro
target row-by-row:              loc263/264/266 still drift
target normal-shape microbatch: loc263/264/266/267 all exact
```

This target should turn that debug-only oracle into a real target-verify MoE
runtime contract and validate the full exactness matrix.  Do not hard-code the
current anchors.

TARGET 11.3 graph/perf promotion remains no-go until exactness passes.

## Goal

Implement a non-debug target-verify MoE path that executes flattened verify rows
in normal-shape-compatible microbatches:

```text
target verify rows: [B * W, hidden]
B = live parent target batch size
W = target-verify width / rows per request
execute MoE in W chunks of [B, hidden]
reassemble output to [B * W, hidden]
```

The target passes when one of these is true:

1. The runtime microbatch path makes the focused bs2 anchors exact, passes the
   full `bs=1/2/4/5/6` greedy exactness matrix, and records a lightweight timing
   trade-off.
2. Or it proves a precise no-go: the debug oracle is exact, but a production
   runtime version cannot be implemented safely without a larger MoE backend or
   scheduler contract change.
3. Or it fixes the current layer0 owner but exposes a new first owner, with
   enough evidence for the next focused target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_layer0_moe_row_shape_precision_contract/README.md
performance_milestones/target11_mtp_target_verify_layer0_moe_row_shape_precision_contract/raw/
performance_milestones/target11_mtp_target_verify_layer0_moe_output_subboundary_parity/README.md
prompts/TARGET_11.245_dsv4_sm80_mtp_target_verify_layer0_moe_row_shape_precision_contract.md
prompts/TARGET_11.244_dsv4_sm80_mtp_target_verify_layer0_moe_output_subboundary_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Do not reopen SWA store/commit/restore, layer2 attention read-side state,
logits, sampler, graph/perf, low-precision research, or communication policy.
Do not branch on batch size, uid, event id, depth, rank, token, expert, layer,
loc, or prompt text.
Treat the debug oracle as a correctness proof, not as code to copy blindly into
runtime without validating metadata, reduce, and perf trade-offs.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/distributed/impl.py
```

Likely implementation points:

```text
_dsv4_is_target_verify_batch
DeepseekV2MoE.forward
target_verify_row_invariant_local
apply_experts_row_invariant
apply_experts / fused runner path
moe_route_dispatch_bf16_marlin_wna16_prepacked
moe_route_dispatch_bf16_marlin_wna16
post-expert all-reduce / reduce dtype
dsv4_target_verify_metadata row mapping / width / parent batch size
```

SGLang reference, for contract comparison and wording:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v2.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
```

## Non-Goals

- Do not start CUDA graph capture, graph replay tuning, or throughput
  optimization beyond a lightweight timing warning.
- Do not change SWA/cache lifecycle.
- Do not patch final norm, lm_head, sampler, C4/C128, PyNCCL, or communication
  policy.
- Do not special-case anchors `263/264/266/267`.
- Do not special-case `bs=2`, `bs=6`, or a fixed verify width.
- Do not promote a layer-only special case unless the report proves a generic
  path is unsafe and writes a follow-up plan.

## Work Plan

### 1. Reproduce The Oracle Baseline

Reproduce TARGET 11.245's focused oracle:

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
```

Confirm before runtime changes:

```text
loc263/264/266: current target MoE drifts
loc267: current target MoE exact
normal-shape microbatch oracle: loc263/264/266/267 exact
```

### 2. Define Runtime Metadata Contract

Identify how to compute:

```text
B: live parent target batch size
W: verify rows per request / fixed verify width
row order: request-major or depth-major
chunk slices that correspond to normal-shape-compatible [B, hidden] batches
active mask for each chunk
handling of rejected tail / bonus rows
```

Write a small table in the report:

```text
metadata field
source object
expected value for bs2 focused repro
expected value for bs6 full-matrix guard
fallback behavior if unavailable
```

Fail closed if `B` or `W` cannot be derived generically.

### 3. Implement The Runtime Microbatch Path

Implement a real target-verify MoE runtime path:

```text
if target_verify and microbatch contract enabled:
    reshape/slice flattened rows into W chunks of B rows
    execute the same MoE path on each chunk
    reassemble output in original flattened row order
else:
    use existing path
```

Important constraints:

```text
Preserve original flattened row order.
Preserve dtype and device.
Preserve all-reduce semantics.
Do not detach or copy through CPU.
Do not use debug oracle output as runtime output.
Avoid per-token/loc/uid/layer special branches.
```

Implementation can start as opt-in if needed, but the report must state the
intended promotion path after validation.

### 4. Validate Focused Anchors

Rerun focused bs2 checks:

```text
loc263/264/266/267 layer0 MoE sub-boundary trace
layer0.moe_output exactness
layer0.post_moe_residual exactness
layer2 SWA store/read trace from TARGET 11.242 or equivalent
layer2 attention split from TARGET 11.241 or equivalent
visible bs2 output exactness
```

If layer0 MoE becomes exact but another boundary fails, close with the new first
owner and do not start graph/perf.

### 5. Full Matrix And bs6 Guard

Run the full matrix:

```text
bs=1
bs=2
bs=4
bs=5
bs=6 in the full 1/2/4/5/6 schedule
```

The bs6 guard must stay in full-matrix mode:

```text
bs6 req5 token6
target-verify input [361, 582, 2067]
target [582, 77296, 3362]
draft [582, 2067]
accepted_prefix=1
mismatch_depth=1
out_cache_loc [265, 266, 267]
```

Do not rely on isolated bs6 if it becomes exact.

### 6. Lightweight Timing And Risk Ledger

Record a lightweight timing warning, not a full perf campaign:

```text
number of MoE calls before/after in target verify
rough per-target-verify MoE time before/after, if available
expected overhead as W microbatches per MoE layer
whether overhead is only on accepted target-verify path
whether CUDA graph support would need a separate plan
```

If overhead is large, keep the path opt-in and write a follow-up optimization
target.  Correctness still comes first.

### 7. SGLang / Mini Contract Note

Write a short source-parity note:

```text
SGLang source forwards full verify rows through MoE, but Mini's active SM80
marlin_wna16 backend is row-shape sensitive.
Mini's microbatch contract is therefore a backend-specific correctness guard,
not a claim that SGLang uses the same shape.
```

If a better SGLang backend port would eliminate the need for microbatching,
record that as a future performance target, not this correctness fix.

### 8. Minimal Fix Policy

Allowed:

```text
generic target-verify MoE microbatching by parent batch size B
opt-in env or config flag while validating
small helper to compute B/W from target-verify metadata
focused debug hooks to prove order and exactness
```

Forbidden:

```text
branch on loc 263/264/266/267
branch on bs2 or bs6
branch on expert id, token id, uid, event, depth, rank, layer, or prompt text
overwrite MoE output from debug oracle
disable accepted commit
patch SWA store/commit/read again
```

## Validation

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

Runtime checks:

```text
focused bs2 layer0 MoE exactness
focused bs2 producer/SWA/layer2 attention guard
bs=1/2/4/5/6 exactness matrix
accepted commit stats
bs6 full-matrix guard
lightweight timing warning
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_target_verify_moe_normal_shape_microbatch_runtime/README.md
```

The README must include:

```text
summary verdict
implementation summary
runtime metadata contract table
exactness matrix before/after
accepted commit stats
focused bs2 MoE/SWA/layer2 guard
bs6 full-matrix guard
lightweight timing and risk ledger
SGLang/Mini contract note
promotion recommendation: default / opt-in / no-go
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The production microbatch path cannot derive `B`, `W`, or row order
  generically from target-verify metadata.
- Runtime microbatch output does not match the debug oracle.
- Layer0 MoE becomes exact but the visible matrix still fails with a new first
  owner; close with that owner.
- The path only works by branching on batch size, uid, event, depth, rank,
  token, expert, layer, loc, or prompt text.
- The path passes correctness but has high overhead; keep it opt-in and write a
  performance/backend target rather than promoting blindly.

# TARGET 11.24: DSV4 SM80 MTP Post-Layer1 / Logits Owner Census

## Status

Next after TARGET 11.23.

TARGET 11.23 closed the scoped layer0 `wo_b` projection/reduce owner for the
original `bs=4 uid0 event4 depth0/depth1` anchor:

```text
layer0.attention_wo_a_output: exact
layer0.attention_wo_b_local_output_before_reduce: exact
layer0.attention_wo_b_post_all_reduce_output: exact
layer0.final_attention_output: exact
layer0.post_moe_residual: exact
layer1.input: exact
layer1.kv_after_kv_norm_rope: exact
```

The broad exactness matrix still fails only at:

```text
bs=2 req1 token6: baseline 7557, MTP 13097
bs=6 req5 token6: baseline 2067, MTP 77296
```

A focused `bs=2` trace showed the first failing row is still exact through the
traced layer1 KV boundary:

```text
inputs: [6102, 621]
positions: [10, 12]
row: uid1 / row1
embedding -> layer0 -> layer0 post-MoE -> layer1.input -> layer1 KV: exact
```

Therefore the next owner is no longer layer0 `wo_a`, layer0 `wo_b`, layer0 MoE,
or the already traced layer1 KV producer boundary.  This target should locate
the first remaining owner after layer1, or prove that hidden states are exact
and the drift is in logits, sampler row mapping, or accepted-commit bookkeeping.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the first boundary where baseline greedy and MTP target-verify/accepted
commit diverge for the remaining failing rows, starting from the known exact
layer1 boundary and ending at final logits/top1.

The target passes when it produces one of these:

1. A precise first hidden-state owner after layer1, with layer id, sub-boundary,
   rank/depth/row identity, and before/after exactness counts.
2. Proof that hidden states remain exact through final norm/lm_head input, but
   logits or `lm_head` all-gather diverge.
3. Proof that logits/top1 are exact, but sampler row selection, token commit,
   or visible-output bookkeeping diverges.
4. A precise no-go explaining which required row identity or instrumentation is
   missing.

This target is primarily an attribution target.  Do not start graph/perf work,
and do not apply a broad fix unless the first owner is already unambiguous and
the fix is small, source-aligned, and validated across the matrix.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer0_wo_b_projection_reduce_parity/README.md
performance_milestones/target11_mtp_layer0_wo_b_projection_reduce_parity/raw/
prompts/TARGET_11.23_dsv4_sm80_mtp_layer0_wo_b_projection_reduce_parity.md
prompts/TARGET_11.22_dsv4_sm80_mtp_layer0_wo_a_projection_contract_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward these constraints:

```text
Do not reopen wo_a/wo_b unless new evidence proves regression.
Do not reopen layer0 MoE unless the new first mismatch points there.
Do not patch indexer FP8, C128, SWA lifecycle, or communication before the
first post-layer1 owner is known.
Do not branch on batch size, uid, event id, depth, rank, token, or prompt text.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/distributed/impl.py
```

SGLang references:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Use SGLang as the reference contract for target-verify metadata, row/depth
packing, final hidden/logits preparation, and accepted-row commit semantics. If
Mini intentionally differs, prove the difference is exact for
baseline-vs-target rows before keeping it.

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not add per-batch, per-request, per-token, per-layer, or per-rank special
  cases.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not undo TARGET 11.22 `wo_a` or TARGET 11.23 `wo_b` fixes.
- Do not make low-precision, FP8 KV cache, PyNCCL, or communication-policy
  changes.
- Do not run heavy macro experiments before the first owner is localized.

## Work Plan

### 1. Reproduce The Remaining Matrix

Use the same matrix and environment as TARGET 11.23 unless the report explains
why a change is required:

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

Confirm the matrix before deeper instrumentation:

```text
bs=1 exact
bs=2 fail at req1 token6, or explain movement
bs=4 exact
bs=5 exact
bs=6 fail at req5 token6, or explain movement
```

If the failing rows move, follow the new first visible mismatch and record why
the old anchors are stale.

### 2. Row Identity And Commit-State Sanity

For the first failing `bs=2` row and, if cheap, the first failing `bs=6` row,
record both baseline and MTP identities:

```text
uid / request index
visible token index
event id
target-verify row id
row depth
row category: accepted draft / correction / bonus / normal target decode
input ids for the active row group
positions
seq lens before and after commit
component loc ids
SWA loc ids, if active
compressed/C128 state ids, if active
```

This is a guard against comparing the right tensor value on the wrong logical
row. If row identity is inconsistent, stop with a scheduler/commit-row owner
instead of continuing operator bisection.

### 3. Coarse Layer Boundary Census After Layer1

Start from the known exact boundary and perform a coarse bisection through the
remaining target layers.

Prefer low-volume checkpoints first:

```text
layerN.input
layerN.final_attention_output
layerN.post_moe_residual or layerN.output
final_norm_input
final_norm_output
lm_head local logits
lm_head gathered logits
sampler selected top1
visible committed token
```

Do not instrument every sub-boundary in every layer on the first pass.  Use
coarse bisection to locate the first bad layer or final-logits boundary, then
expand only around that owner.

For every compared checkpoint, include:

```text
layer / boundary name
rank
row identity
shape / dtype / stride / storage offset / contiguity
hash
max_delta / mean_delta
first differing index
exact rows count
```

### 4. Owner Classification

Classify the first non-exact boundary into one of these buckets:

```text
hidden_state_owner:
    A layer/submodule after layer1 first changes hidden states.

logits_owner:
    Hidden states are exact through final norm or lm_head input, but local or
    gathered logits differ.

sampler_owner:
    Logits/top1 are exact, but sampler row selection or visible output differs.

commit_bookkeeping_owner:
    Computed tensors are exact, but accepted/correction/bonus rows are committed
    to the wrong logical request/token position.

instrumentation_no_go:
    The current debug payload cannot compare the same row/state across baseline
    and MTP.
```

The README must explicitly state which bucket won and why.

### 5. Source-Parity Table

Before any fix, write a compact source-parity table against SGLang for the
winning owner.  Include only relevant rows, but cover at least:

```text
target-verify row/depth packing
layer forward mode for target verify
component/SWA/C128 metadata used by the first bad boundary
final norm and lm_head input row selection
lm_head gather/reduce semantics
sampler selected row and token commit semantics
```

### 6. Minimal Fix Policy

A small fix is allowed only if the owner is precise and the fix aligns Mini with
SGLang or with Mini's normal target decode contract.

Allowed examples:

```text
wrong row/depth mapping -> align row identity with target-verify contract
wrong final logits row selection -> use the same row mapping as baseline/SGLang
local hidden boundary uses a target-only branch -> align with normal decode path
```

Forbidden fixes:

```text
branch on bs=2 or bs=6
branch on req id, uid, event id, token id, layer number, rank, or prompt text
overwrite final tokens after sampling
disable accepted commit to pass exactness
change graph/perf/communication/low-precision paths
```

### 7. Validation

After attribution, and after any small fix if attempted, rerun:

```text
bs=1/2/4/5/6 exactness matrix
accepted commit stats
focused bs=2 first-failing-row trace
focused bs=6 first-failing-row trace if bs=6 still fails or changed
TARGET 11.22 wo_a sanity or equivalent focused checkpoint
TARGET 11.23 wo_b sanity or equivalent focused checkpoint
```

Minimum static checks:

```bash
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/distributed/impl.py

git diff --check
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_post_layer1_logits_owner_census/README.md
```

The README must include:

```text
summary verdict
implementation/instrumentation summary
before/after exactness matrix if any fix is attempted
accepted commit stats
row identity and commit-state sanity table
coarse layer boundary census after layer1
first owner classification
source-parity table against SGLang for the winning owner
wo_a/wo_b regression sanity
first remaining owner or promotion/no-go verdict
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The first failing row cannot be mapped to the same logical baseline-vs-MTP
  request/token/position/depth.
- The first mismatch is found after layer1; close this target with that exact
  owner instead of chasing unrelated operators.
- Hidden states are exact through final norm but logits differ; close with a
  logits/lm_head owner target recommendation.
- Logits/top1 are exact but visible tokens differ; close with a sampler or
  commit-bookkeeping owner target recommendation.
- A proposed fix only passes by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt content.
- The matrix still fails after a safe fix; close with the new first owner rather
  than starting graph/perf.

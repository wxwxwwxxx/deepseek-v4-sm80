# TARGET 11.18: DSV4 SM80 MTP Post-MoE Downstream Owner Census

## Status

Next after TARGET 11.17.

TARGET 11.17 fixed the scoped MoE pre-reduce drifting-rank owner:

```text
bs=2 event0 layer0 rank0 expert_aggregate_before_reduce: exact after fix
bs=1 event0 layer7 rank0 expert_aggregate_before_reduce: exact after fix
bs=1 event0 layer7 rank7 expert_aggregate_before_reduce: exact after fix
```

The MoE culprit was target-verify multi-row local MoE staging before aggregate:

```text
bs=2 rank0/layer0: routed_expert_output_raw drift
bs=1 rank0/layer7: routed_expert_output_raw drift
bs=1 rank7/layer7: shared_expert_output_raw tiny drift
```

The local MoE fix is scoped and should be kept. Full MTP promotion is still
no-go:

```text
11.17 exactness matrix:
  bs=1 pass
  bs=2 fail
  bs=4 fail
  bs=5 fail
  bs=6 fail
```

New remaining owners reported by TARGET 11.17:

```text
bs=2 focused after-fix:
  first row0 mismatch: layer21.attention_wo_b_post_all_reduce_output
  max_abs_delta = 0.03125

bs=4 row0 owner run:
  event0 and event1 row0 parity exact
  event2 first mismatch: layer32.indexer_query_fp8_values
  later events also expose attention / wkv / wo_a / wo_b boundaries
```

This target should not immediately chase one of those in isolation. First build
a post-MoE downstream owner census and rank the remaining owners by causal
priority.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Establish the next correctness target after the MoE fix by producing a
source-aligned, event/layer/rank-aware census of downstream target-verify
owners.

The target passes when one of these is true:

1. It identifies a single highest-priority downstream owner, with enough
   boundary evidence for the next target to fix it.
2. Or it proves a small source-aligned fix for that owner and improves the
   exactness matrix without regressing the scoped MoE fix.

The target must explicitly decide how to prioritize:

```text
bs=2 layer21 attention_wo_b_post_all_reduce_output
bs=4 event2 layer32 indexer_query_fp8_values
later attention / wkv / wo_a / wo_b owners
```

Do not start CUDA graph, throughput profiling, acceptance tuning, or broad MTP
runtime rewrites in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
performance_milestones/target11_mtp_moe_post_reduce_parity/README.md
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
prompts/TARGET_11.17_dsv4_sm80_mtp_moe_pre_reduce_drifting_rank_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Important interpretation:

```text
Layer21/layer32 does not necessarily mean error slowly accumulates until those
layers. It means all earlier observed boundaries were equivalent for that
event, then the first currently visible non-equivalent downstream owner appears
there. Later events may expose different owners because accepted commit,
metadata, KV/component state, and target-verify rows have changed.
```

Also keep this debugging rule:

```text
Single-kernel bit-exact/allclose is still useful for local owner attribution,
but MTP correctness now also requires event-level state, accepted commit,
rank-local/all-rank communication, and final exactness matrix checks.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
python/minisgl/distributed/impl.py
performance_milestones/target11_mtp_spec_runtime_v1/scripts/spec_runtime_exactness.py
```

Relevant Mini boundaries:

```text
layer input / residual state
attention q path
indexer_query_fp8_values
C4/C128 attention metadata and cache reads
wkv / attention output
attention_wo_a
attention_wo_b local output
attention_wo_b_post_all_reduce_output
moe_input / moe_output sanity only
lm/head row0 logits
accepted commit state
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
```

Relevant SGLang behavior to inspect and cite:

```text
target-verify metadata preparation
frozen-KV read-only draft/verify behavior
online C128 pending/write/commit behavior
indexer query FP8 quant/dequant or backend dispatch
attention projection and post-all-reduce staging
target-verify row/depth mapping across accepted commit events
```

Use SGLang source behavior as the preferred contract. If Mini intentionally
differs, prove the Mini path is exact for normal-vs-target required rows.

## Non-Goals

- Do not undo or weaken the TARGET 11.17 MoE row-invariant local fix.
- Do not add parent batch size, active verify length, request slot, rank id,
  layer id, token id, or prompt-content special branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not start graph/perf work.
- Do not fix indexer FP8 or `attention_wo_b` blindly before ranking owners in a
  shared census.
- Do not treat one operator allclose result as sufficient if the input state or
  accepted commit state differs.

## Work Plan

### 1. Reproduce The Post-11.17 Matrix

Use the same correctness contract:

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
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
```

Reproduce:

```text
bs=1 pass
bs=2 fail
bs=4 fail
bs=5 fail
bs=6 fail
```

If this no longer reproduces, rerun the broad row0 owner trace and follow the
new first owner. Do not debug stale layer numbers.

### 2. Build An Event/Lifecycle Census

For each failing batch size, record:

```text
request id
event index
decode token index
draft tokens proposed
target verify rows
accepted draft tokens
target correction rows
target_verify_rows_committed
accepted_kv_copied_tokens
whether row0 is accepted/correction/bonus/tail if available
first visible token diff
first row0 hidden/logit diff
```

At minimum, include:

```text
bs=2 failing event around layer21 attention_wo_b_post_all_reduce_output
bs=4 event0, event1, event2 through layer32 indexer_query_fp8_values
bs=5 and bs=6 first failing event, if cheap
```

The goal is to distinguish:

```text
operator compute drift with exact inputs
state/metadata drift where operator inputs already differ
collective propagation from another rank
accepted-commit lifecycle drift that appears only after event N
```

### 3. Coarse Layer Owner Timeline

Before splitting either owner deeply, produce a table:

```text
batch size
event
request / visible row
first exact layer
first drifting layer
first drifting boundary
rank-local or all-rank
input exact?
output allclose?
max delta
likely category: operator / metadata-state / communication / commit-lifecycle
```

Cover these candidate boundaries:

```text
layer input / residual
attention q path
indexer_query_fp8_values
wkv / attention output
attention_wo_a
attention_wo_b local output
attention_wo_b_post_all_reduce_output
moe_input / moe_output
final row0 logits
```

The table should answer whether `bs=2 layer21 attention_wo_b` and
`bs=4 layer32 indexer FP8` are independent owners or two symptoms of an earlier
shared state/metadata mismatch.

### 4. Focused Split: bs=2 Layer21 Attention wo_b

Only after the coarse table confirms this owner is still relevant, split:

```text
attention input hidden
q/indexer inputs
attention output before wo_a/wo_b
wo_b input
wo_b local matmul output
wo_b post-all-reduce output
final_attention_output / residual output
```

Record:

```text
shape / dtype / stride / storage offset / contiguity
input exactness
rank-local contribution exactness
all-rank contribution if post-all-reduce drifts
backend and communication label
BF16/FP32 cast order
first differing index and sample values
```

If the input to `wo_b` is already non-exact, do not fix `wo_b`; trace backward
to the input owner.

### 5. Focused Split: bs=4 Event2 Layer32 Indexer FP8

Only after the coarse table confirms this owner is still relevant, split:

```text
layer input hidden
indexer input
indexer pre-quant value
indexer quantized FP8 value
indexer dequant/consumer value, if available
attention metadata consumed by indexer
C4/C128/component/cache state used by the event
```

Record:

```text
input exactness
FP8 scale / dtype / backend if available
row/depth mapping
cache/page/component loc metadata
accepted commit state before event2
whether event0/event1 exactness changes the state
```

If the indexer input is already non-exact, do not fix indexer FP8; trace back to
the input or commit-state owner.

### 6. Source-Parity Table

Write a table before any fix:

```text
Concept
SGLang behavior
Mini normal decode
Mini target-verify
Verdict / action
```

Cover at least:

```text
target-verify row/depth mapping
accepted commit state before later events
attention metadata preparation
indexer FP8 quant/dequant boundary
wo_b projection and post-all-reduce staging
online C128 pending/write/commit if it affects the owner
```

### 7. Prioritize The Next Fix

The report must rank candidate next targets. Prefer the owner that:

```text
appears earliest in a causal event/layer trace
can explain multiple failing batch sizes
has exact inputs and non-exact outputs, or a clearly identified state mismatch
matches a known SGLang contract Mini does not yet implement
can be fixed without batch/rank/layer/token special casing
```

If there is no clear single owner, close with a precise no-go and propose the
smallest next census, not a broad rewrite.

### 8. Optional Minimal Fix

If the census proves one simple source-aligned owner, a minimal fix is allowed.
Validation must include:

```text
the original owner trace
the full bs=1/2/4/5/6 exactness matrix
MoE pre-reduce sanity from TARGET 11.17
accepted commit stats
```

Do not promote MTP or start graph/perf even if one downstream owner is fixed.

## Validation Gates

Minimum validation:

```text
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/distributed/impl.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/engine/engine.py

git diff --check
```

Matrix:

```text
bs=1
bs=2
bs=4
bs=5
bs=6
```

Use the same six fixed prompts from TARGET 11.15-11.17 when `bs=6` is included.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_post_moe_downstream_owner_census/README.md
```

The README must include:

```text
summary verdict
implementation summary
post-11.17 exactness matrix
event/lifecycle census
coarse layer owner timeline
focused attention_wo_b evidence, if relevant
focused indexer FP8 evidence, if relevant
SGLang source-parity table
owner priority ranking
accepted commit stats
remaining owner or minimal-fix result
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The post-11.17 matrix no longer reproduces and the first owner moved.
- `bs=2 layer21 attention_wo_b` or `bs=4 layer32 indexer FP8` has non-exact
  inputs, because the true owner is earlier state/metadata rather than that
  local operator.
- The evidence shows a SGLang target-verify metadata/commit contract is missing
  and the fix is larger than one focused operator.
- A proposed fix passes only by branching on batch/rank/layer/event/token
  identity.
- One candidate owner is fixed but the matrix still fails; close with the next
  first owner instead of starting graph/perf.

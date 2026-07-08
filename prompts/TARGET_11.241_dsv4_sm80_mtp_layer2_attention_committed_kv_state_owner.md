# TARGET 11.241: DSV4 SM80 MTP Layer2 Attention Committed-KV State Owner

## Status

Next after TARGET 11.24.

TARGET 11.24 reduced the remaining MTP correctness problem to a hidden-state
owner inside layer2 attention:

```text
bs=2 req1 token6: baseline 7557, MTP 13097
embedding -> layer0 -> layer1 -> layer2.input: exact on all 8 ranks
layer2.final_attention_output: first mismatch, 0/8 exact
final_norm / lm_head / sampler: downstream propagation
```

TARGET 11.24 also showed the `bs=6` failure is context-sensitive:

```text
bs=6 alone can become exact
bs=6 inside the full 1/2/4/5/6 matrix reproduces req5 token6 drift
```

This target should split layer2 attention into current-token compute versus
consumed committed cache/state.  The primary anchor is the deterministic `bs=2`
failure.  The `bs=6` full-matrix failure must be included as a secondary
cross-case/lifecycle guard, but it should not make the first pass branch into a
separate investigation unless it contradicts the bs=2 owner.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the first layer2 attention sub-boundary where baseline greedy and MTP
target/accepted-commit state diverge for the remaining failure.

The target should answer:

```text
Given exact layer2.input for the same token/position, does layer2 attention drift
because the current Q path differs, or because it consumes different committed
KV/SWA/C128/page-table state?
```

The target passes when it produces one of these:

1. `current_q_owner`: q/q_norm/RoPE/projection inputs differ before attention
   reads cache.
2. `committed_kv_owner`: current Q is exact, but consumed K/V or compressed
   state differs.
3. `metadata_owner`: Q and consumed cache values are exact, but page/slot,
   seq-len, mask, C4/C128/SWA metadata, or attention row mapping differs.
4. `attention_kernel_owner`: Q, cache values, and metadata are equivalent, but
   the attention backend output differs.
5. `instrumentation_no_go`: the current debug hooks cannot compare the same
   layer2 row/cache state across baseline and MTP.

If a minimal, source-aligned fix is obvious after the first owner is proven, it
may be attempted.  Otherwise close with the owner and next repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_post_layer1_logits_owner_census/README.md
performance_milestones/target11_mtp_post_layer1_logits_owner_census/raw/
performance_milestones/target11_mtp_layer0_wo_b_projection_reduce_parity/README.md
prompts/TARGET_11.24_dsv4_sm80_mtp_post_layer1_logits_owner_census.md
prompts/TARGET_11.23_dsv4_sm80_mtp_layer0_wo_b_projection_reduce_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Do not reopen layer0 wo_a/wo_b/MoE unless a new trace proves regression.
Do not treat lm_head/sampler as first owner; 11.24 showed drift starts earlier.
Do not branch on batch size, uid, event id, depth, rank, token, or prompt text.
Use SGLang as the reference for target-verify metadata and cache-state
lifecycle whenever Mini behavior is ambiguous.
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

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Source-parity focus:

```text
target-verify prepare/commit state visible to layer2
req_to_token / page table / slot mapping after accepted commit
SWA tail or full-cache rows visible to layer2
C4/C128 compressed state and online C128 MTP pending/commit behavior
attention metadata masks/seq_lens for normal target decode after MTP commits
```

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not patch final norm, lm_head, sampler, or visible-output formatting.
- Do not change low-precision, FP8 KV cache, PyNCCL, or communication policy.
- Do not disable accepted commit or fail closed just to recover exactness.
- Do not add special cases for `bs=2`, `bs=6`, uid, event, depth, rank, layer,
  token, or prompt text.
- Do not broaden into all-layer instrumentation before layer2 attention is
  split.

## Work Plan

### 1. Reproduce Anchors

Use the same environment as TARGET 11.24:

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

Confirm:

```text
bs=2 req1 token6: layer2.input exact, layer2.final_attention_output drifts
full 1/2/4/5/6 matrix: bs2 and bs6 still fail, unless the report explains why
```

For `bs=6`, preserve the full-matrix reproduction mode.  Do not rely only on an
isolated `bs=6` run if it becomes exact.

### 2. Layer2 Current-Q Path Split

For the bs=2 primary anchor, compare baseline vs MTP on all ranks:

```text
layer2.input
layer2 attention input/pre-norm, if distinct
q_a / q_a_norm / q_b input
q_nope / q_pe / q_norm_rope output
kv_a / kv_a_norm / compressed current-token path, if produced for the row
attention query tensor passed to the attention backend
```

Record for each:

```text
rank
shape / dtype / stride / storage offset / contiguity
hash
max_delta / mean_delta
first differing index
exact ranks count
backend path
```

If the current Q path drifts before any cache read, stop with
`current_q_owner` and recommend the next sub-boundary repair.

### 3. Layer2 Consumed Cache/State Split

If the current Q path is exact, compare the committed state consumed by layer2
attention:

```text
req_to_token entries for the sequence
page ids / slot ids / token positions
seq_lens and effective attention lengths
SWA locs and retained tail rows, if layer2 uses SWA
compressed KV / C128 state locs
K/V or compressed-cache values read by the attention backend
metadata masks and active rows
```

Prefer small sampled state first:

```text
last few committed tokens before the failing position
all accepted/correction rows from the preceding MTP verify event
rows touched by layer2 for the failing request
```

Then expand only if the sampled rows are exact but layer2 attention still
drifts.

### 4. Attention Backend Boundary

If Q, consumed cache values, and metadata are all exact, instrument the layer2
attention backend boundary:

```text
attention backend selected
input query hash
input cache/value metadata hash
pre-wo merged attention output
after inverse RoPE / projection-prepared output, if applicable
wo_a / wo_b local/reduce sanity only if the first mismatch reaches that far
```

This step should determine whether the owner is the attention kernel/backend
itself or an upstream metadata/cache mismatch that was missed.

### 5. bs6 Cross-Case Guard

Include the `bs=6` full-matrix failure as a lifecycle guard:

```text
full 1/2/4/5/6 matrix bs6 req5 token6
input [361, 582, 2067], target [582, 77296, 3362], draft [582, 2067]
accepted_prefix=1, mismatch_depth=1
```

The bs6 guard should answer:

```text
Does bs6 hit the same owner class as bs2?
Does bs6 only reproduce after prior matrix cases because cache/component/state
from earlier runs changes allocator or lifecycle behavior?
```

If bs6 contradicts bs2, close with a two-owner report and recommend the smaller
next target.  Do not force one fix to cover both without evidence.

### 6. Source-Parity Table

Before any fix, write a compact SGLang source-parity table for the winning
owner:

```text
Concept
SGLang behavior
Mini baseline normal decode
Mini MTP target/accepted-commit path
Candidate fix or no-go
```

Cover at least:

```text
target-verify accepted-row commit into target KV/component state
layer2 attention metadata preparation after accepted commit
SWA and C128 state visible to layer2
page/slot/req_to_token mapping
attention backend dispatch and input tensors
```

### 7. Minimal Fix Policy

A fix is allowed only when the first owner is precise and the implementation
matches SGLang or Mini normal-target semantics.

Allowed examples:

```text
accepted commit skipped or wrote wrong layer/state -> commit the correct state
metadata points layer2 to stale slot/page -> align metadata lifecycle
C128/SWA pending state missing for accepted rows -> port the SGLang-equivalent
pending/write/commit behavior for that state
```

Forbidden examples:

```text
special-case bs2 or bs6
overwrite layer2 attention output
disable accepted commit
force isolated bs6 behavior to replace full-matrix behavior
patch lm_head/sampler before layer2 attention is exact
```

## Validation

After attribution, and after any minimal fix if attempted:

```text
bs=1/2/4/5/6 exactness matrix
focused bs=2 layer2 attention split trace
full-matrix bs6 guard trace
accepted commit stats
TARGET 11.23 wo_b sanity or equivalent checkpoint
TARGET 11.24 post-layer1 coarse census or equivalent checkpoint
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
performance_milestones/target11_mtp_layer2_attention_committed_kv_state_owner/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
bs2 primary anchor layer2 attention split
bs6 full-matrix lifecycle guard
current-Q vs consumed-cache/state verdict
SGLang source-parity table
first owner classification
wo_a/wo_b regression sanity
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The bs=2 first failure no longer reproduces and no stable replacement anchor
  is identified.
- The current Q path differs before cache consumption; close with
  `current_q_owner`.
- Q is exact but consumed cache/state or metadata differs; close with
  `committed_kv_owner` or `metadata_owner`.
- Q/cache/metadata are exact but attention backend output differs; close with
  `attention_kernel_owner`.
- bs6 requires full-matrix history to reproduce but the target cannot preserve
  that context; document this as an instrumentation blocker.
- Any proposed fix only works by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt text.
- A safe fix improves bs2 but regresses bs4/bs5 or changes the established
  layer0->layer1 exactness; close with the regression details.

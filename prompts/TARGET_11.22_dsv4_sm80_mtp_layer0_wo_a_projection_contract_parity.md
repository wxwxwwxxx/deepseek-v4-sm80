# TARGET 11.22: DSV4 SM80 MTP Layer0 `wo_a` Projection Contract Parity

## Status

Next after TARGET 11.21.

TARGET 11.21 found the first producer-side mismatch feeding the bad
`swa.layer1` committed row values:

```text
bs=4 uid0 event4 depth0 token 582:
  first propagated baseline-vs-MTP mismatch: layer0.attention_wo_a_output

bs=4 uid0 event4 depth1 correction token 9628:
  first propagated baseline-vs-MTP mismatch: layer0.attention_wo_a_output
```

Across all `8 TP ranks x 2 depths`:

```text
layer0.merged_attention_output_before_wo: exact 16/16
layer0.merged_attention_output_after_inverse_rope: exact 16/16
layer0.attention_wo_a_output: exact 0/16
```

This means the `swa.layer1` value mismatch is fed by layer0 `wo_a` projection
contract mismatch, not by:

```text
row ownership
page table / loc mapping
seq_len / position ownership
layer0 attention/KV read-write
layer1 SWA store/copy
C128
indexer FP8
```

TARGET 11.21's source-parity evidence:

```text
Mini baseline normal decode uses _wo_a_bf16_bmm_projection.
Mini target verify uses _wo_a_bf16_bmm_projection_row_invariant.
SGLang DSV4 applies wo_a over the actual T rows; no target-verify-only
row-invariant wo_a branch was found in the inspected source.
```

Therefore TARGET 11.22 should align Mini's target-verify `wo_a` projection
contract with baseline/SGLang behavior, or prove the exact contract needed.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Make layer0 `attention_wo_a_output` equivalent between baseline greedy and MTP
target-verify for the exact same
`merged_attention_output_after_inverse_rope` rows, or prove the precise
projection contract gap that must be fixed next.

The target passes when one of these is true:

1. `layer0.attention_wo_a_output` becomes exact for the `bs=4 uid0 event4`
   depth0/depth1 anchor rows, the downstream `swa.layer1` mismatch improves or
   closes, and the exactness matrix improves without regressing prior fixes.
2. Or the target produces a precise no-go naming the exact `wo_a` projection
   backend/dtype/layout/order mismatch and a safe next implementation plan.

The target should answer:

```text
Given identical layer0 merged attention rows after inverse RoPE, why do
baseline greedy and target-verify produce different wo_a outputs?
```

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_row_depth_producer_parity/README.md
performance_milestones/target11_mtp_row_depth_committed_state_baseline_parity/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
prompts/TARGET_11.21_dsv4_sm80_mtp_target_verify_row_depth_producer_parity.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Important TARGET 11.21 constraints:

```text
Do not repair attention_wo_b, attention_wo_a downstream reduce, indexer FP8,
C128, layer1 SWA store, or page table before proving the wo_a projection
contract.

Do not reopen lifecycle ownership: TARGET 11.20 proved row-depth ownership is
equivalent and value mismatch starts at producer output.

Keep q-path raw hash drift secondary unless it becomes output-significant before
merged_attention_output_before_wo.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
performance_milestones/target11_mtp_target_verify_row_depth_producer_parity/scripts/summarize_producer_trace.py
```

Likely Mini code paths:

```text
_wo_a_bf16_bmm_projection
_wo_a_bf16_bmm_projection_row_invariant
target-verify is_target_verify branch around wo_a
cached BF16 wo_a weight / layout path
normal decode actual-row projection path
row-invariant local/reduce helpers
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Relevant SGLang behavior to inspect and cite:

```text
wo_a projection in DeepSeek V4 attention
target-verify row tensor shape around wo_a
whether SGLang uses actual-row projection or any target-verify special branch
dtype/backend used for wo_a on SM80
```

Use SGLang source behavior as the preferred contract. If Mini intentionally
differs, prove Mini's contract is bit-exact for baseline-vs-target rows.

## Non-Goals

- Do not start graph/perf work.
- Do not patch `attention_wo_b`, indexer FP8, C128, MoE, or layer1 SWA store.
- Do not undo the TARGET 11.17 MoE row-invariant local fix.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not add parent batch size, uid, event id, row depth, rank, layer, token, or
  prompt-content special branches.
- Do not accept allclose if BF16 hash/exactness still drives downstream
  `swa.layer1` mismatch.

## Work Plan

### 1. Reproduce The Anchor

Use the TARGET 11.20/11.21 authoritative contract:

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

Reproduce:

```text
bs=4 uid0 event4 depth0/depth1
layer0.merged_attention_output_after_inverse_rope exact
layer0.attention_wo_a_output mismatch 16/16 rank-depth rows
downstream swa.layer1 mismatch
```

If the anchor moves, follow the new first producer mismatch and explain why the
old one is stale.

### 2. Build A Direct `wo_a` Projection Micro-Or Oracle

For the exact same `merged_attention_output_after_inverse_rope` rows from the
anchor, run or compare:

```text
Mini baseline actual-row _wo_a_bf16_bmm_projection
Mini target-verify _wo_a_bf16_bmm_projection_row_invariant
candidate actual-row target-verify projection
candidate row-invariant projection used by both paths, if tested
SGLang-style torch.einsum / actual-T-row reference, if cheap
```

Record:

```text
input hash
weight hash / layout id if available
input shape / dtype / stride / storage offset / contiguity
backend path
output shape / dtype / stride
output row hash
max_delta / mean_delta
first differing index
rank/depth
```

The immediate goal is to determine whether the mismatch is caused by:

```text
row-invariant target-verify projection path
weight/layout/view difference
batched vs single-row accumulation order
BF16/FP32 cast order
einsum/BMM shape semantics
padding/active row treatment
```

### 3. Source-Parity Contract Decision

Write a source-parity table before any fix:

```text
Concept
SGLang behavior
Mini baseline normal decode
Mini MTP target verify
Candidate fix
Verdict
```

Cover at least:

```text
wo_a input tensor shape
actual-row vs row-invariant projection
dtype / accumulation behavior
weight layout
target-verify row-depth packing
padding rows
post-wo_a output shape consumed by wo_b
```

Preferred contract:

```text
target verify should use the same source-aligned projection semantics as
baseline/SGLang for the actual verify rows.
```

Only keep a row-invariant path if it is proven bit-exact against that contract
for all anchor rank-depth rows and does not rely on batch/rank/depth special
cases.

### 4. Implement A Minimal Source-Aligned Fix

Allowed fix candidates:

```text
make target verify use the baseline actual-row wo_a projection path
replace both baseline and target verify with a shared source-aligned projection
adjust only dtype/layout/cast order if the oracle proves that is the mismatch
```

Forbidden fixes:

```text
branch on bs=4 / uid0 / event4 / depth0/depth1 / rank / layer0
post-hoc overwrite layer1 SWA rows
disable accepted commit
skip wo_a checks by changing trace/compare logic
```

### 5. Validate Downstream Closure

After any fix, validate:

```text
layer0.attention_wo_a_output exact for bs=4 uid0 event4 depth0/depth1
layer0.final_attention_output improves/closes
layer0.post_moe_residual improves/closes
layer1.input improves/closes
layer1.kv_after_kv_norm_rope / swa.layer1 improves/closes
event8 pre-verify swa.layer1 improves/closes
```

Then rerun:

```text
full bs=1/2/4/5/6 exactness matrix
accepted commit stats
TARGET 11.17 MoE pre-reduce sanity or an equivalent focused sanity
```

If `wo_a` closes but the matrix still fails, close this target with the new
first owner instead of starting graph/perf.

## Validation Gates

Minimum static checks:

```text
python -m py_compile \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/utils/dsv4_mtp_debug.py \
  python/minisgl/distributed/impl.py

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

Use the same six prompts from TARGET 11.20/11.21 unless the report states why a
new prompt set is required.

Focused gates:

```text
bs=4 uid0 event4 depth0/depth1 wo_a projection oracle
bs=4 uid0 event4 layer0->layer1 downstream timeline
bs=4 uid0 event8 pre-verify swa.layer1 comparison
```

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer0_wo_a_projection_contract_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
direct wo_a projection oracle
source-parity table against SGLang
chosen projection contract
before/after layer0->layer1 timeline
MoE sanity result
first remaining owner or promotion/no-go verdict
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The `swa.layer1` anchor moves and `layer0.attention_wo_a_output` is no longer
  the first output-significant producer mismatch.
- The direct `wo_a` oracle cannot compare baseline vs target-verify on the same
  input rows; document the missing instrumentation.
- The fix would require a broad projection backend rewrite; document the exact
  contract and split the implementation into the next target.
- A proposed fix passes only by branching on batch size, uid, event, depth,
  rank, layer, token, or prompt content.
- `wo_a` becomes exact but `swa.layer1` remains mismatched; close with the new
  first downstream owner.
- The exactness matrix still fails after a safe fix; close with the new first
  owner rather than starting graph/perf.

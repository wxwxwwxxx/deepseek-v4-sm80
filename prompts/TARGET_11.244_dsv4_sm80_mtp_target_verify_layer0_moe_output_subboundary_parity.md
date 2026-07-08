# TARGET 11.244: DSV4 SM80 MTP Target-Verify Layer0 MoE Output Sub-Boundary Parity

## Status

Next after TARGET 11.243.

TARGET 11.243 proved that the bad target-verify writer rows are equivalent to
baseline normal writer rows through layer0 attention, and first become
output-significantly non-equivalent at layer0 MoE output:

```text
classification: layer0_owner
subtype: layer0.moe_output producer owner

bad loc 263: event0 depth0 accepted, token/pos 1275 / 7
bad loc 264: event0 depth1 correction, token/pos 2353 / 8
bad loc 266: event1 depth0 accepted, token/pos 2693 / 10
control loc 267: event1 depth1 correction, token/pos 751 / 11
```

Current exactness for the four anchors:

```text
embedding: 8/8 for all anchors
layer0.input: 8/8 for all anchors
layer0.final_attention_output: 8/8 for all anchors
layer0.post_attention_residual: 8/8 for all anchors
layer0.moe_input: 8/8 for all anchors
layer0.moe_output: 0/8 for loc263/264/266, 8/8 for loc267
layer0.post_moe_residual: 0/8 for loc263/264/266, 8/8 for loc267
```

The transient q-side 7/8 mismatch seen in TARGET 11.243 is not the current
owner because loc267 has the same transient q-side blip but reconverges by
layer0 attention output and remains exact through layer2 input.

This target should split only the layer0 MoE path for the current anchors.  Do
not reopen SWA store/commit/restore, layer2 attention read-side state, logits,
sampler, or graph/perf unless new evidence proves the owner moved.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find the first layer0 MoE sub-boundary where baseline normal writer rows and
MTP target-verify writer rows diverge for the current bad anchors, while the
loc267 control remains exact.

The target should answer:

```text
Given exact layer0.moe_input, why do loc263/264/266 produce non-equivalent
layer0.moe_output while loc267 stays exact?
```

The target passes when it produces one of these classifications:

1. `router_owner`: router logits, topk ids, topk weights, or normalization
   differ for bad rows.
2. `routed_expert_owner`: routing is equivalent, but routed expert output
   differs.
3. `shared_expert_owner`: routed expert path is equivalent, but shared expert
   output differs.
4. `aggregation_owner`: routed/shared expert outputs are equivalent, but
   aggregation, gating weights, residual combination, or row scatter differs.
5. `reduce_owner`: pre-reduce aggregate is equivalent, but post-reduce/all-
   reduce MoE output differs.
6. `target_verify_row_shape_owner`: target verify uses a multi-row MoE shape,
   padding, or row ordering that changes computation relative to baseline
   normal decode, while a row-wise oracle is exact.
7. `backend_precision_owner`: the active MoE backend is not row-shape stable
   under the current target-verify shape/precision.
8. `instrumentation_no_go`: current debug hooks cannot compare the same MoE
   sub-boundary across baseline and MTP.

If a minimal source-aligned fix is clear after the owner is proven, it may be
attempted.  Otherwise close with the owner and a smaller repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_target_verify_layer2_input_producer_parity/README.md
performance_milestones/target11_mtp_target_verify_layer2_input_producer_parity/raw/
performance_milestones/target11_mtp_layer2_swa_commit_state_producer_owner/README.md
prompts/TARGET_11.243_dsv4_sm80_mtp_target_verify_layer2_input_producer_parity.md
prompts/TARGET_11.242_dsv4_sm80_mtp_layer2_swa_commit_state_producer_owner.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Historical MoE reference, useful but not authoritative for current anchors:

```text
prompts/TARGET_11.15_dsv4_sm80_mtp_moe_output_subboundary_parity.md
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
performance_milestones/target11_mtp_moe_post_reduce_parity/README.md
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
```

Carry forward:

```text
Do not patch SWA store, snapshot restore, committed restore, later read,
layer2 attention read-side cache, C4, logits, sampler, or graph/perf.
Do not branch on batch size, uid, event id, depth, rank, token, layer, expert,
loc, or prompt text.
Use SGLang MoE behavior as the reference when Mini normal decode and target
verify differ.
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

Likely Mini MoE paths:

```text
router/gate input
router logits
topk ids / topk weights
routed expert dispatch
Marlin WNA16 or cached BF16 expert backend
shared expert path
routed + shared aggregation
expert reduce / all-reduce
moe_output
post_moe_residual
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
/workspace/sglang-main/python/sglang/srt/layers/moe/utils.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
```

Source-parity focus:

```text
MoE row batch shape and active-row mask in target verify
router dtype and topk normalization
routed expert dispatch order and padding
expert backend precision and row-shape stability
shared expert execution and aggregation order
post-expert reduce dtype/op/backend
target-verify rows that are accepted/correction/rejected tail
```

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not patch SWA lifecycle; 11.242 ruled it out as the first owner.
- Do not patch layer2 attention read-side state; 11.241 traced it back to the
  SWA values produced upstream.
- Do not patch q-path transient mismatch unless it becomes output-significant.
- Do not patch final norm, lm_head, sampler, C4/C128, low precision, PyNCCL, or
  communication policy.
- Do not special-case locs `263/264/266/267`.
- Do not special-case `bs=2` or `bs=6`.
- Do not promote a slow reference MoE path as runtime unless it is explicitly
  marked as a temporary correctness oracle.

## Work Plan

### 1. Reproduce Current MoE Anchors

Use the same environment as TARGET 11.243:

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
loc263/264/266: layer0.moe_input exact, layer0.moe_output drifts
loc267 control: layer0.moe_input exact, layer0.moe_output exact
```

If anchors move, follow the new bad/control rows and explain why the old rows
are stale.

### 2. Row Identity And MoE Context Table

Before comparing MoE tensors, write a compact table:

```text
loc
uid / event / depth
row category: accepted / correction / bonus / rejected tail
input token / position
target token / draft token
target-verify group shape
row index inside target-verify tensor
active mask / padded rows
normal decode row shape
target-verify row shape
MoE backend selected
expert backend selected
```

This table should make it clear whether bad rows and the loc267 control differ
in row depth, row category, tensor position, target-verify group shape, or
backend path.

### 3. Layer0 MoE Sub-Boundary Census

For locs `263/264/266/267`, compare baseline normal writer against MTP
target-verify writer at:

```text
moe_input
router_input, if distinct
router_logits
topk_ids
topk_weights
topk_weights_after_norm, if distinct
routed_expert_input
routed_expert_output
shared_expert_input
shared_expert_output
expert_aggregate_before_reduce
expert_reduce_output / post_all_reduce, if applicable
moe_output
post_moe_residual
```

Record for each boundary:

```text
rank
shape / dtype / stride / storage offset / contiguity
hash
allclose / bit-exact if cheap
max_delta / mean_delta
first differing index
expert ids and weights for each row
active token/expert counts
backend path
```

For sparse/routed data, explicitly record row/expert ordering so an apparent
value mismatch is not just a packing mismatch.

### 4. Router And TopK Oracle

If router/topk differs:

```text
Compare router input hashes against moe_input.
Compare router dtype and accumulation dtype.
Compare topk ids, topk weights, normalized weights, and selected expert order.
Run a row-wise router oracle for the same target-verify rows.
Check whether padded/rejected rows contaminate topk selection.
```

Close with `router_owner` if this is the first mismatch.

### 5. Routed Expert Oracle

If routing is equivalent but routed expert output differs:

```text
Compare routed expert input after dispatch.
Compare per-expert token counts/order/padding.
Compare expert backend path for normal decode vs target verify.
Run a row-wise routed expert oracle for bad rows and loc267 control.
If Marlin WNA16/cached BF16 backend is active, test whether target-verify
multi-row shape changes result relative to normal row shape.
```

Close with `routed_expert_owner`, `target_verify_row_shape_owner`, or
`backend_precision_owner` as appropriate.

### 6. Shared Expert Oracle

If routed expert output is equivalent but shared expert output differs:

```text
Compare shared expert input.
Compare shared expert activation path.
Compare shared expert projection backend and dtype.
Run a row-wise shared expert oracle if target verify uses a different batch
shape.
```

Close with `shared_expert_owner` if this is the first mismatch.

### 7. Aggregation / Reduce Oracle

If routed and shared expert outputs are equivalent:

```text
Compare gated weighted sum before reduce.
Compare routed + shared aggregation order.
Compare post-expert reduce input and output.
Compare reduce dtype/op/backend and label.
Compare local per-rank aggregate before all-reduce.
```

Close with `aggregation_owner` or `reduce_owner` if this is the first mismatch.

### 8. Row-Shape Stability Oracle

Because TARGET 11.23 already found row-shape-sensitive projection behavior,
run a diagnostic row-shape oracle if any MoE backend mismatch appears:

```text
normal decode row shape
target-verify full group shape
target-verify active rows only
target-verify row-by-row reference
```

This oracle is diagnostic only.  A slow row-by-row path may be used to prove
correctness, but should not be promoted unless a separate performance plan is
written.

### 9. bs6 Full-Matrix Guard

Keep the full matrix guard from TARGET 11.242/11.243:

```text
bs6 req5 token6
target-verify input [361, 582, 2067]
target [582, 77296, 3362]
draft [582, 2067]
accepted_prefix=1
mismatch_depth=1
out_cache_loc [265, 266, 267]
```

The bs6 guard should answer:

```text
Does bs6 show the same layer0 MoE sub-boundary owner as bs2?
Does bs6 require full-matrix history only for visibility, or also for the MoE
sub-boundary to reproduce?
```

Do not require a full bs6 fix unless it contradicts bs2.

### 10. SGLang Source-Parity Table

Before any fix, write a compact source-parity table:

```text
Concept
SGLang behavior
Mini baseline normal decode
Mini MTP target verify
Candidate fix / no-go
```

Cover at least:

```text
MoE row batch shape and active rows
router/topk dtype and normalization
routed expert dispatch and padding
expert backend precision
shared expert path
routed/shared aggregation
post-expert reduce/all-reduce
target-verify accepted/correction/rejected rows
```

### 11. Minimal Fix Policy

A fix is allowed only if the first owner is precise and source-aligned.

Allowed examples:

```text
router/topk dtype/shape mismatch -> align with normal decode/SGLang
padded/rejected rows contaminate routing or aggregation -> exclude inactive rows
expert backend row-shape instability -> canonicalize active row shape or use a
row-stable backend for target verify
reduce dtype/order mismatch -> align with normal decode/SGLang
```

Forbidden examples:

```text
special-case loc 263/264/266/267
special-case bs2 or bs6
special-case expert ids or token ids
overwrite moe_output from a reference path at runtime
disable accepted commit
patch SWA store/commit/read after 11.242 ruled it out
patch final sampled token directly
```

## Validation

After attribution, and after any minimal fix if attempted:

```text
bs=1/2/4/5/6 exactness matrix
focused bs=2 layer0 MoE sub-boundary trace for locs 263/264/266/267
focused bs=2 producer-boundary trace from TARGET 11.243 or equivalent
focused bs=2 layer2 SWA store/read trace from TARGET 11.242 or equivalent
full-matrix bs6 MoE guard
accepted commit stats
```

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
performance_milestones/target11_mtp_target_verify_layer0_moe_output_subboundary_parity/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
row identity and MoE context table
layer0 MoE sub-boundary census
router/topk oracle
routed expert oracle
shared expert oracle
aggregation/reduce oracle
row-shape stability oracle, if needed
bs6 full-matrix guard
SGLang source-parity table
first owner classification
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- `layer0.moe_input` is no longer exact for the anchors; return to producer
  boundary attribution.
- The first MoE sub-boundary owner is found; close with that owner instead of
  continuing broad MoE/perf experimentation.
- The MoE mismatch is caused by target-verify row-shape/backend instability and
  the safe fix requires a larger backend-port target.
- A proposed fix only works by branching on batch size, uid, event, depth, rank,
  token, expert, layer, loc, or prompt text.
- A safe fix improves bs2 but regresses bs4/bs5, established SWA store/commit
  sanity, or layer2 attention read-side controls.

# TARGET 11.243: DSV4 SM80 MTP Target-Verify Layer2 Input Producer Parity

## Status

Next after TARGET 11.242.

TARGET 11.242 proved that the layer2 SWA cache/store/commit path faithfully
preserves the value it receives, but bad rows are already non-equivalent before
`store_cache`:

```text
classification: producer_value_owner

bs=2 req1 token6 later reads bad layer2 SWA locs:
  loc 263: target-verify accepted row, token/pos 1275 / 7, store input 0/8 exact
  loc 264: target-verify correction row, token/pos 2353 / 8, store input 0/8 exact
  loc 266: target-verify accepted row, token/pos 2693 / 10, store input 0/8 exact

control:
  loc 267: target-verify correction row, token/pos 751 / 11, store input 8/8 exact
```

The bad rows are non-equivalent already at:

```text
layer2.input
layer2.attention_input
layer2.wkv_output
layer2.kv_after_kv_norm_rope
layer2 SWA store input
```

The store path is not the culprit:

```text
store input == immediate cache after store: 8/8
snapshot restore preserves given value
committed restore preserves given value
later read consumes the same committed value
```

This target should find where the target-verify writer hidden state first
diverges before `layer2.input`.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Compare baseline normal writer rows against MTP target-verify writer rows for
the same logical token/position, and find the first producer boundary where the
bad writer rows become non-equivalent.

The target should answer:

```text
Why do target-verify rows for locs 263/264/266 enter layer2 already
non-equivalent, while the neighboring correction control loc267 is exact?
```

The target passes when it produces one of these classifications:

1. `row_identity_owner`: the compared baseline and target-verify writer rows
   are not actually the same logical token/position/prefix state.
2. `embedding_or_input_owner`: token/position embedding or first hidden input
   differs before layer0.
3. `layer0_owner`: rows are exact into layer0, but diverge inside layer0.
4. `layer1_owner`: rows are exact through layer0, but diverge inside layer1.
5. `layer2_boundary_owner`: rows are exact through layer1 output but differ
   when building `layer2.input` or entering layer2.
6. `target_verify_dependency_owner`: the bad target-verify rows depend on an
   earlier target-verify row in the same flattened verify group whose state is
   non-equivalent to baseline greedy.
7. `instrumentation_no_go`: current debug hooks cannot compare the same writer
   row across baseline and MTP.

If a minimal source-aligned fix is obvious after the first owner is proven, it
may be attempted.  Otherwise close with the owner and a narrower repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer2_swa_commit_state_producer_owner/README.md
performance_milestones/target11_mtp_layer2_swa_commit_state_producer_owner/raw/
performance_milestones/target11_mtp_layer2_attention_committed_kv_state_owner/README.md
prompts/TARGET_11.242_dsv4_sm80_mtp_layer2_swa_commit_state_producer_owner.md
prompts/TARGET_11.241_dsv4_sm80_mtp_layer2_attention_committed_kv_state_owner.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Do not patch SWA store, snapshot restore, committed restore, later read,
attention backend, C4 cache, wo_a/wo_b, logits, or sampler unless new evidence
proves regression.
Do not branch on batch size, uid, event id, depth, rank, token, layer, loc, or
prompt text.
Use SGLang target-verify row/depth dependency and metadata preparation as the
reference contract.
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/utils/dsv4_mtp_debug.py
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
```

Source-parity focus:

```text
target-verify row/depth packing
flattened verify group dependencies between rows
accepted/correction/bonus row hidden-state selection
front-chain hidden states passed to the target model
target-verify positions / seq_lens / req_to_token state
layer input construction for target-verify rows
```

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not patch SWA store/commit/restore; 11.242 ruled them out as the first
  owner for the current failure.
- Do not patch final norm, lm_head, sampler, C4, C128, low-precision, PyNCCL,
  or communication policy.
- Do not special-case locs `263/264/266/267`; they are debug anchors.
- Do not special-case `bs=2` or `bs=6`.
- Do not broaden into all-layer performance profiling.

## Work Plan

### 1. Reproduce Writer Anchors

Use the same environment as TARGET 11.242:

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

Confirm the writer anchors:

```text
bad loc 263: uid1 depth0 event0 accepted, token/pos 1275 / 7
bad loc 264: uid1 depth1 event0 correction, token/pos 2353 / 8
bad loc 266: uid1 depth0 event1 accepted, token/pos 2693 / 10
control loc 267: uid1 depth1 event1 correction, token/pos 751 / 11
```

If anchors move, follow the new bad/control rows and explain why the old rows
are stale.

### 2. Row Identity And Dependency Contract

Before comparing tensors, write a row identity table:

```text
loc
uid / request index
event id
row depth
row category: accepted / correction / bonus / rejected tail
input token
target token
draft token, if any
position
seq_len before verify
seq_len for the target-verify row
baseline normal decode prefix length
target-verify group input tokens
target-verify group positions
```

Then classify each row's dependency:

```text
independent_from_same_verify_group:
    row should depend only on committed prefix before verify.

depends_on_prior_verify_row:
    row should consume hidden/KV/state produced by an earlier row in the same
    flattened verify group.

correction_or_bonus_tail:
    row is selected after mismatch and may have different dependency semantics.
```

This is important because bad rows and exact controls may differ in whether they
are allowed to depend on prior target-verify rows.

### 3. Coarse Producer Boundary Census

For bad locs `263/264/266` and control loc `267`, compare baseline normal writer
against MTP target-verify writer at:

```text
embedding / initial hidden
layer0.input
layer0.final_attention_output
layer0.post_moe_residual
layer1.input
layer1.final_attention_output
layer1.post_moe_residual
layer2.input
```

Record:

```text
rank
shape / dtype / stride / storage offset / contiguity
hash
max_delta / mean_delta
first differing index
exact ranks count
row identity
```

The first pass should be coarse.  Do not split layer0/layer1 internals until
the first bad layer is known.

### 4. Intra-Layer Split At The First Bad Layer

Once the coarse census finds the first bad layer, split only that layer.

For an attention-side owner, compare:

```text
attention input / norm
q path
kv/current path
indexer metadata
consumed SWA/C4/C128 cache values
merged attention output before wo
wo_a / wo_b only if drift first appears there
```

For a MoE-side owner, compare:

```text
moe input
router/topk
routed expert output
shared expert output
aggregate before reduce
post-reduce output
post-MoE residual
```

For a residual/layer-boundary owner, compare:

```text
residual input
attention output
MoE output
residual add order / dtype
next layer input construction
```

### 5. Target-Verify Dependency Oracle

If the first bad row depends on a prior target-verify row, build or reuse a
small oracle to answer:

```text
Does Mini target verify feed row depth d with the same hidden/KV/component state
that baseline greedy would have after committing rows < d?
```

Compare at least:

```text
event0 depth0 -> event0 depth1
event1 depth0 -> event1 depth1
bad accepted rows versus exact correction control
```

If Mini is using stale committed-prefix state where SGLang expects updated
front-chain state, or vice versa, close with `target_verify_dependency_owner`.

### 6. bs6 Full-Matrix Guard

Preserve the full `1/2/4/5/6` matrix guard:

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
Does bs6 show the same producer-boundary owner as bs2?
Does bs6 expose a target-verify dependency pattern that bs2 also has?
```

Do not require a full bs6 fix unless it contradicts bs2.

### 7. SGLang Source-Parity Table

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
row/depth packing and row dependency
positions and seq_lens
target-verify front-chain hidden state
accepted/correction/bonus row selection
layer input construction
cache/component state visible between rows in the same verify group
```

### 8. Minimal Fix Policy

A fix is allowed only if the first owner is precise and source-aligned.

Allowed examples:

```text
target-verify depth d uses stale row input -> feed the correct previous verify
state according to SGLang contract
row/depth dependency metadata wrong -> align metadata with SGLang
layer input construction skips committed front-chain row -> include it
```

Forbidden examples:

```text
special-case loc 263/264/266/267
special-case bs2 or bs6
force target-verify writer values from baseline traces
patch SWA cache/store/commit after 11.242 ruled them out
disable accepted commit
patch final sampled token directly
```

## Validation

After attribution, and after any minimal fix if attempted:

```text
bs=1/2/4/5/6 exactness matrix
focused bs=2 producer-boundary trace for locs 263/264/266/267
focused bs=2 layer2 SWA store/read trace from TARGET 11.242 or equivalent
full-matrix bs6 producer guard
TARGET 11.241 layer2 attention split or equivalent checkpoint
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
performance_milestones/target11_mtp_target_verify_layer2_input_producer_parity/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
row identity and dependency contract table
coarse producer boundary census
intra-layer split for the first bad layer, if reached
target-verify dependency oracle, if needed
bs6 full-matrix guard
SGLang source-parity table
first owner classification
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- Writer row identity cannot be aligned between baseline normal decode and MTP
  target verify.
- The first bad boundary is found; close with that owner instead of continuing
  broad all-layer instrumentation.
- A target-verify dependency mismatch is proven; close with the dependency
  contract and a repair recommendation.
- A proposed fix only works by branching on batch size, uid, event, depth, rank,
  layer, token, loc, or prompt text.
- A safe fix improves bs2 but regresses bs4/bs5, established layer2 attention
  read-side exact controls, or SWA store/commit sanity.

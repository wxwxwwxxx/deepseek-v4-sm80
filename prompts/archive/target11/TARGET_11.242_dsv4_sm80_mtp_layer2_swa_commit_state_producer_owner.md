# TARGET 11.242: DSV4 SM80 MTP Layer2 SWA Commit-State Producer Owner

## Status

Next after TARGET 11.241.

TARGET 11.241 classified the remaining first owner as
`committed_kv_owner`, specifically layer2 SWA cache/state consumed by layer2
attention:

```text
bs=2 req1 token6: baseline 7557, MTP 13097
layer2.input: exact
layer2 current Q path: exact
layer2 current-token KV/compressed path: exact
metadata / page / slot / seq_len / mask: exact
attention backend: splitk on both sides
consumed C4 compressed cache: exact
consumed SWA cache values: non-equivalent
layer2.attention_backend.merged_attention_output_before_wo: first drift
```

For the primary `bs=2` anchor, rank0 consumed SWA rows showed a small set of
bad historical rows:

```text
active SWA full locs: [268, 267, 266, 265, 264, 263, 262, 261, ...]
loc 268: exact current row
loc 267: exact
loc 266: drift
loc 265: exact
loc 264: drift
loc 263: drift
loc 262: exact
loc 261: exact
```

This target must determine whether those bad layer2 SWA rows are:

```text
produced incorrectly by target verify / normal producer;
stored incorrectly into the SWA cache;
correct after store but later overwritten/restored/copied incorrectly;
or compared through an invalid row/loc mapping.
```

The `bs=6` full-matrix failure remains a secondary lifecycle guard, because it
only reliably reproduces in the full `1/2/4/5/6` schedule.

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Trace the producer-to-consumer lifecycle of the layer2 SWA rows later consumed
by the failing attention read, and identify the first point where baseline
greedy and MTP accepted-commit state become non-equivalent.

The target passes when it produces one of these classifications:

1. `producer_value_owner`: the value passed to layer2 SWA store is already
   non-equivalent to the baseline committed row.
2. `store_cache_owner`: the store input is equivalent, but the SWA cache value
   immediately after store differs.
3. `commit_restore_owner`: the value is correct after store, but becomes wrong
   during target-verify snapshot restore, accepted-row commit, or post-commit
   state promotion.
4. `overwrite_lifetime_owner`: the value is correct after commit, but later
   overwritten by another row/request/layer lifecycle event before the failing
   read.
5. `mapping_owner`: the compared loc is not the same logical token/position
   across baseline and MTP, despite 11.241's read-side metadata appearing
   equal.
6. `instrumentation_no_go`: the write/read provenance cannot yet be recovered
   from existing debug hooks.

If a minimal SGLang-aligned fix is clear after the owner is proven, it may be
attempted.  Otherwise close with the owner and a smaller repair target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_layer2_attention_committed_kv_state_owner/README.md
performance_milestones/target11_mtp_layer2_attention_committed_kv_state_owner/raw/
performance_milestones/target11_mtp_post_layer1_logits_owner_census/README.md
prompts/TARGET_11.241_dsv4_sm80_mtp_layer2_attention_committed_kv_state_owner.md
prompts/TARGET_11.24_dsv4_sm80_mtp_post_layer1_logits_owner_census.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Carry forward:

```text
Do not reopen Q path, C4 cache, wo_a/wo_b, layer0 MoE, logits, or sampler unless
new evidence proves regression.
Do not branch on batch size, uid, event id, depth, rank, token, layer, loc, or
prompt text.
Use SGLang as the reference for SWA store/commit/snapshot semantics whenever
Mini behavior is ambiguous.
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
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
```

Source-parity focus:

```text
get_swa_out_cache_loc / full-to-SWA loc translation
store_cache inputs and output locs for layer2 SWA
target-verify temporary rows versus committed rows
accepted/correction/bonus row commit rules
snapshot restore before/after target verify
req_to_token updates and SWA eviction/update after accepted commit
```

## Non-Goals

- Do not start CUDA graph or throughput optimization.
- Do not patch lm_head, sampler, final norm, C4, C128, or low-precision paths.
- Do not change PyNCCL or communication policy.
- Do not disable accepted commit or fail closed to recover exactness.
- Do not special-case `loc 263/264/266`; they are debug anchors, not feature
  branches.
- Do not broaden into all-layer SWA lifecycle until layer2's bad locs have a
  producer/commit owner.

## Work Plan

### 1. Reproduce Read-Side Owner

Use the same environment as TARGET 11.241:

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

Confirm for the primary anchor:

```text
bs=2 req1 token6
layer2.input exact
layer2 Q path exact
SWA active locs exact as indices
SWA consumed values drift at locs including 263/264/266
merged_attention_output_before_wo drifts
```

If the bad loc set changes, follow the new bad locs and explain why the old
ones are stale.

### 2. Build A Layer2 SWA Loc Provenance Table

For every bad consumed SWA loc in the primary anchor, recover:

```text
loc id
logical request / uid
token id
position
writer event id
writer row id / row depth
writer row category: baseline normal decode, target verify accepted row,
correction row, bonus row, normal target decode after commit
layer id
rank
```

For the first pass, prioritize:

```text
bs=2 req1 locs 266, 264, 263
neighbor exact locs 267, 265, 262 as controls
rank0 plus one additional rank if cheap, then all ranks after owner appears
```

If provenance cannot be recovered, stop with `instrumentation_no_go` and
describe the missing hook.

### 3. Producer-Value Trace

For each bad loc and one neighboring exact control, compare baseline writer row
versus MTP writer row at:

```text
layer2 writer input
layer2 attention input
layer2 wkv_output
layer2 kv_after_kv_norm_rope
layer2 SWA store input tensor/value
get_swa_out_cache_loc output
```

Record:

```text
shape / dtype / stride / storage offset / contiguity
hash
max_delta / mean_delta
first differing index
row identity
rank
```

If SWA store input is already non-equivalent, close with
`producer_value_owner`.  The next repair should focus on the writer row's
producer path, not on SWA store/commit.

### 4. Store And Immediate Cache Trace

If store input is equivalent, compare:

```text
SWA cache value before store at target loc
SWA cache value immediately after store
store output loc / translated SWA loc
any aliasing between full loc and SWA loc
```

If the store input is exact but immediate cache differs, close with
`store_cache_owner`.

### 5. Commit / Restore / Snapshot Lifecycle Trace

If immediate store is exact, trace the same loc through:

```text
pre-target-verify snapshot
after target-verify temporary write
after snapshot restore for rejected rows
after accepted/correction/bonus commit
after req_to_token update
before the later normal producer read
at the failing layer2 attention read
```

Classify:

```text
commit_restore_owner:
    correct after store but wrong after restore/commit/promotion.

overwrite_lifetime_owner:
    correct after commit but overwritten by a later unrelated row/request/layer
    before the failing read.

mapping_owner:
    the loc points to a different logical token/position across baseline and
    MTP.
```

### 6. bs6 Full-Matrix Lifecycle Guard

Keep the full `1/2/4/5/6` reproduction for bs6:

```text
bs6 req5 token6
input [361, 582, 2067]
target [582, 77296, 3362]
draft [582, 2067]
accepted_prefix=1
mismatch_depth=1
out_cache_loc [265, 266, 267]
c128_pending_write_commit=ready
```

The bs6 guard should answer:

```text
Does bs6 show the same owner class as bs2?
Does full-matrix history change which SWA locs are bad?
Does any prior batch in the same Python process alter allocator/component/SWA
lifetime state before bs6?
```

Do not require a full bs6 source fix unless bs2 is already explained and bs6
contradicts it.

### 7. SGLang Source-Parity Table

Before any fix, write a compact source-parity table:

```text
Concept
SGLang behavior
Mini baseline normal decode
Mini MTP target/accepted-commit path
Candidate fix / no-go
```

Cover at least:

```text
SWA out loc generation
SWA store input and value semantics
target-verify temporary writes
accepted/correction/bonus row commit
snapshot restore for rejected rows
req_to_token and sequence length updates
SWA eviction/update behavior after commit
```

### 8. Minimal Fix Policy

A fix is allowed only if the first owner is precise and the behavior matches
SGLang or Mini normal-target semantics.

Allowed examples:

```text
target-verify accepted SWA row is not committed -> commit the correct SWA row
snapshot restore clobbers an accepted row -> restore only rejected/temp rows
wrong full-to-SWA loc used at commit -> align loc translation with read path
writer row category misclassified -> align accepted/correction/bonus ownership
```

Forbidden examples:

```text
special-case loc 263/264/266
special-case bs2 or bs6
overwrite SWA cache at read time
copy baseline values into MTP debug path
disable accepted commit
patch attention output directly
```

## Validation

After attribution, and after any minimal fix if attempted:

```text
bs=1/2/4/5/6 exactness matrix
focused bs=2 SWA producer/commit/read trace
full-matrix bs6 lifecycle guard
TARGET 11.241 layer2 attention split or equivalent checkpoint
TARGET 11.23 wo_b sanity or equivalent checkpoint
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
performance_milestones/target11_mtp_layer2_swa_commit_state_producer_owner/README.md
```

The README must include:

```text
summary verdict
instrumentation / implementation summary
exactness matrix before/after if any fix is attempted
accepted commit stats
bs2 bad-loc provenance table
producer-value trace
store/immediate-cache trace
commit/restore/snapshot lifecycle trace
bs6 full-matrix lifecycle guard
SGLang source-parity table
first owner classification
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The bad consumed SWA locs cannot be mapped to writer rows.
- The writer row identity differs between baseline and MTP; close with
  `mapping_owner`.
- Store input is already non-equivalent; close with `producer_value_owner`.
- Store input is exact but immediate cache differs; close with
  `store_cache_owner`.
- Immediate cache is exact but later committed/read cache differs; close with
  `commit_restore_owner` or `overwrite_lifetime_owner`.
- A proposed fix only works by branching on batch size, uid, event, depth, rank,
  layer, token, loc, or prompt text.
- A safe fix improves bs2 but regresses bs4/bs5 or the established
  layer2-current-Q / C4 / metadata exactness; close with regression details.

# TARGET 11.10: DSV4 SM80 MTP Target-Verify Layer0 Attention/KV Producer Parity

## Status

Next after TARGET 11.9.

TARGET 11.9 made useful structural progress but produced a precise no-go for
promoting the new SGLang-shaped runtime:

```text
default runtime: legacy_target11_6
diagnostic runtime: MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
```

`sglang_prefill_extend` is explicit and runnable with accepted commit enabled,
but it is not exact.  The first non-batch-special-case owner is:

```text
case: bs=1, req0, token index 6
runtime: sglang_prefill_extend
first owner: layer0.merged_attention_output_before_wo
row0 normal oracle vs target-verify row0
max_abs_delta = 0.03125
```

This target should fix or precisely no-go that first owner before any graph or
throughput work resumes.

## Goal

Make `sglang_prefill_extend` target-verify layer0 row0 produce the same
attention/KV result as normal target decode for the same request, prefix,
position, and visible token.

SGLang source parity is a first-class goal in this target, not a supporting
check after experiments.  Prefer porting or adapting SGLang's DeepSeek V4
target-verify metadata/prepare semantics over iterating through many mini-local
probe experiments.  Probes should validate the source-aligned contract; they
should not replace the contract.

The target passes when:

```text
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1 exact
layer0 normal oracle vs target-verify row0 parity passes at the first attention
boundary and after wo/output merge
accepted commit remains enabled
```

Then expand the gate:

```text
bs=1/2/4/5/6 exact, or fail with a new first owner after layer0 attention/KV
producer parity has been proven
```

Do not start CUDA graph, throughput, C128 boundary promotion, or speculative
acceptance tuning in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_sglang_aligned_target_verify_runtime_mode/README.md
performance_milestones/target11_mtp_target_verify_runtime_contract_unification/README.md
prompts/TARGET_11.9_dsv4_sm80_mtp_sglang_aligned_target_verify_runtime_mode.md
```

Important TARGET 11.9 result:

```text
The issue is no longer "which bs special case should be patched".
The explicit SGLang-shaped runtime fails at layer0 attention output for bs=1.
```

Candidate matrix from 11.9:

```text
sglang_prefill_extend base attention + fused KV:
  bs=1 fail [0]
  bs=2 fail [0,1]
  bs=4 fail [3]
  bs=5 pass
  bs=6 fail [0,4]

fused KV + split-k target verify:
  bs=1 pass
  bs=2/4/5/6 fail

fused KV + force-torch:
  bs=1/2/4/5/6 fail
```

Interpretation:

```text
Attention backend choice alone is not the full owner.
KV producer/store and target-verify metadata must be checked as one boundary.
```

## References

Mini:

```text
python/minisgl/engine/engine.py
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/csrc/jit/dsv4_online_c128_mtp.cu
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
/workspace/sglang-main/python/sglang/srt/speculative/frozen_kv_mtp_info.py
/workspace/sglang-main/python/sglang/srt/speculative/eagle_utils.py
/workspace/sglang-main/python/sglang/srt/speculative/spec_utils.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
```

Prefer SGLang's DSV4 target-verify metadata and attention contract when mini
and SGLang differ, unless mini has a measured reason to intentionally diverge.
If reference behavior is copied or adapted from SGLang, keep the implementation
small, isolate mini-specific glue, and preserve any relevant source attribution
or license header requirements.

## Non-Goals

- Do not add a new `if bs == ...` fix.
- Do not make `sglang_prefill_extend` pass by switching back to
  `legacy_target11_6` semantics.
- Do not disable accepted commit.
- Do not sequentially recompute accepted rows as the final runtime.
- Do not begin CUDA graph or performance work.
- Do not spend time on C128 boundary gates until the layer0 attention/KV owner
  is exact for the short bs1 repro.
- Do not run broad empirical sweeps before writing the SGLang source-parity
  table for the failing row.  Experiments should confirm or falsify a concrete
  source-parity hypothesis.

## Work Plan

### 1. Reproduce The First Owner

Run the smallest failing repro from TARGET 11.9:

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
bs=1
```

Confirm the same owner:

```text
layer0.merged_attention_output_before_wo
normal target decode row0 oracle vs target-verify row0
```

If the first owner changes, document the change and follow the new first owner.

### 2. Make SGLang Source Semantics The Contract

Before adding more probes, inspect SGLang's DeepSeek V4 MTP target-verify path
and write down the exact prepare semantics that mini should match.

Build a compact parity table for the exact failing row:

```text
concept
SGLang target-verify behavior
mini sglang_prefill_extend behavior
same/different/unknown
action
```

Focus on:

- how SGLang constructs target-verify `seq_lens`, `extend_lens`, and request
  index arrays;
- whether row0 attends to committed prefix plus itself, and how that self row is
  materialized;
- whether target verify is treated as prefill/extend, decode rows, or a
  DSV4-specific hybrid;
- when KV is written relative to attention consumption;
- what exact data structure SGLang passes to the DSV4 attention backend;
- whether SGLang uses a dedicated MTP metadata class or fields that mini is
  currently only approximating;
- whether current mini fused q/k norm + RoPE + store writes exactly the state
  that attention reads.

If mini only copied field names or tensor shapes from SGLang, port the missing
prepare semantics instead of adding another local workaround.  A direct
mini-owned adaptation of SGLang's target-verify metadata prepare is preferred if
the code boundary is reasonably small.

Required output before runtime fixing:

```text
SGLang source owner for each target-verify metadata field
mini current owner for the same field
same/different verdict
implementation action: already aligned / port / adapt / intentionally diverge
```

Only after this table exists should the target perform broad runtime
experiments.  Small probes that directly verify a table entry are allowed.

### 3. Port Or Adapt The Missing Prepare Semantics First

If the table finds a semantic gap in metadata/prepare behavior, fix that before
debugging lower-level kernels.  Examples:

```text
seq_lens construction
extend_lens construction
row-to-request indices
row depth / causal length
active vs padded row masking
KV write-before-attention ordering
target-verify attention backend dispatch key
```

The preferred implementation is:

- small and explicit;
- owned by the `sglang_prefill_extend` runtime mode;
- shared by all parent batch sizes and active verify lengths;
- easy to disable by returning to `legacy_target11_6`;
- accompanied by a source-parity note that names the SGLang source lines or
  functions that motivated the change.

If the required SGLang port is too large, stop with a precise larger-port plan
instead of creating another local approximation.

### 4. Split Layer0 Into Producer And Consumer Boundaries

After source semantics are aligned, add or reuse debug probes that compare
normal target decode row0 and target-verify row0 at these boundaries:

```text
input token / position / out_loc / table_idx
layer0 hidden input
q / q_nope / q_pe after norm and projection
k/v or compressed KV producer inputs
RoPE-applied k/q components
KV written at target-verify out_loc
KV gathered by the attention consumer
attention metadata: seq_lens, extend_lens, row depth, causal length
attention scores/probabilities if cheap enough
attention output before wo
merged_attention_output_before_wo
post-wo attention output
```

The goal is to classify the remaining owner as one of:

```text
metadata mismatch
KV producer/store mismatch
KV gather/readback mismatch
attention causal/mask mismatch
attention kernel numerical mismatch
normal-oracle construction mismatch
```

Do not jump directly to full-model output tokens while this boundary is still
unclassified.

### 5. Build A Correctness-First Local Oracle

If needed, add a slow debug oracle for the failing row:

```text
normal target decode row0 attention/KV producer
vs target-verify row0 attention/KV producer
```

Acceptable oracle behavior:

- runs only under debug env;
- synchronizes and copies small tensors for comparison;
- uses torch/reference attention for a single layer/row;
- records max/mean abs deltas and first differing element.

Not acceptable as the final fix:

- replacing target verify with sequential normal decode;
- disabling accepted commit;
- branching on bs/request/token to select a passing path.

### 6. Fix The First Owner

The preferred fix is to make `sglang_prefill_extend` use one semantically correct
target-verify attention/KV producer contract.

Possible outcomes:

1. SGLang prepare-semantics gap:
   - port/adapt the missing SGLang target-verify metadata preparation behavior;
   - keep mini-specific glue minimal and centralized in the runtime contract.
2. Metadata bug:
   - fix row depth, causal length, seq lens, extend lens, or row-to-request
     mapping;
   - keep the centralized runtime contract from TARGET 11.9.
3. KV producer/store bug:
   - make target verify and normal target decode use equivalent q/k norm, RoPE,
     and store semantics for active rows;
   - do not select producer semantics by parent batch size or active len.
4. KV gather/readback bug:
   - fix page/table/out_loc mapping so attention consumes the row that was just
     produced.
5. Attention kernel semantic bug:
   - either port the SGLang-equivalent path or keep a correctness-first backend
     while documenting the fast kernel gap for a later performance target.
6. Oracle construction bug:
   - repair the oracle and rerun attribution before touching runtime semantics.

### 7. Validate Incrementally

Minimum after the fix:

```text
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1 exact
layer0 attention/KV parity passes for the previous failing req0/token6 row
accepted_kv_commit_fail_closed = false
target_commit_kv_copies > 0
accepted_kv_copied_tokens > 0
```

Then run:

```text
bs=1/2/4/5/6 exactness matrix
bs=7/8/16 light exposure if bs=1/2/4/5/6 passes
```

If `bs=1` passes but another batch fails, report the new first owner and ensure
it is not a reintroduced parent-batch/active-len numerical branch.

## Success Criteria

Minimum:

```text
first owner is classified below layer0.merged_attention_output_before_wo
the classification is supported by tensor/metadata evidence
no new batch-size special case is introduced
accepted commit remains enabled
```

Full:

```text
SGLang target-verify metadata/prepare source-parity table is complete
sglang_prefill_extend bs=1 exact
layer0 target-verify attention/KV producer parity passes
bs=1/2/4/5/6 exact, or next first owner is precisely identified
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- the only possible fix is another parent batch size, active verify length, or
  request slot branch;
- exactness requires disabling accepted commit;
- the normal oracle is found to be invalid and needs its own target;
- SGLang source behavior cannot be matched without a larger target-verify
  attention backend port;
- mini can only match SGLang by copying field shapes while leaving prepare
  semantics ambiguous;
- layer0 parity passes but full exactness still fails at a new owner that needs
  a separate target.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_layer0_attention_kv_parity/README.md
```

Include:

- reproduction of the TARGET 11.9 first owner;
- source-parity table for SGLang target-verify metadata/prepare semantics;
- any ported/adapted SGLang reference behavior and where it lives in mini;
- layer0 boundary parity table;
- SGLang source-parity table for the failing row;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- decision on whether TARGET 11.3 can start or the next correctness target is
  needed.

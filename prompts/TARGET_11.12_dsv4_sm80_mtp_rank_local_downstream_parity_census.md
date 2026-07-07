# TARGET 11.12: DSV4 SM80 MTP Rank-Local Downstream Parity Census

## Status

Next after TARGET 11.11.

TARGET 11.11 closed the requested `attn.wo_b` projection/reduce owner for the
`sglang_prefill_extend` runtime:

```text
layer0.attention_wo_a_output: exact
layer0.attention_wo_b_local_output_before_reduce: exact
layer0.attention_wo_b_post_all_reduce_output: exact
layer0.final_attention_output: exact
bs=1: exact
accepted commit: enabled and healthy
```

The expanded matrix still fails:

```text
bs=1 pass
bs=2 fail req0
bs=4 fail req3
bs=5 fail req0
bs=6 fail req0/3/5
```

The next evidence is no longer a single `wo_b` owner.  It is a rank-local
downstream parity class:

```text
bs=1 next hidden owner: layer2.indexer_query_fp8_values, uint8 max_delta=1.0
bs=2 event0: rank6 upstream drift around layer0 q_after_q_norm_rope / local wo_b
bs=2 event1: layer0.moe_output drifts while layer0.moe_input is exact
bs=2 event2: rank2 upstream drift around layer1 attention / local wo_b
```

This target should classify that downstream class before applying another local
fix.

## Goal

Build a rank-local downstream parity census for `sglang_prefill_extend` after
the 11.11 `wo_b` fix, then fix or precisely no-go the first common owner.

The target passes when one of these is true:

1. A single common mechanism is identified and fixed, followed by:

```text
MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend
bs=1/2/4/5/6 exact
accepted commit remains enabled
```

2. Or, if there are multiple independent owners, the report gives a ranked
repair plan with enough evidence that the next target can focus on one owner
without reopening metadata, attention/KV, or `wo_b`.

Do not start CUDA graph, throughput tuning, C128 boundary gates, or speculative
acceptance tuning in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_wo_b_projection_reduce_parity/README.md
performance_milestones/target11_mtp_layer0_attention_kv_parity/README.md
prompts/TARGET_11.11_dsv4_sm80_mtp_attn_wo_b_projection_reduce_parity.md
```

Important TARGET 11.11 result:

```text
Not owners for the bs=1 first row anymore:
  target-verify metadata
  KV producer/store
  attention consumer
  wo_a
  wo_b local projection
  wo_b all-reduce

New owners:
  indexer FP8 path at layer2 for bs=1
  rank-local upstream drift for bs=2
  MoE output drift in one bs=2 verify event
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/csrc/jit/dsv4_online_c128_mtp.cu
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/online_c128_mtp.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh
```

Use SGLang source behavior as the contract when mini's target-verify downstream
path differs, especially for indexer/compressor state and MoE target-verify row
handling.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id, or
  observed token numerical branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6` as the fix.
- Do not reopen target-verify metadata, attention/KV, or `wo_b` unless new
  evidence proves the previous attribution was invalid.
- Do not optimize graph/perf or run large serving benchmarks.
- Do not let this become an unbounded layer-by-layer hunt.  The main deliverable
  is an owner census and ranked repair decision.

## Work Plan

### 1. Reproduce The 11.11 State

Run or reuse the same required contract:

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

Confirm:

```text
bs=1 exact
layer0 wo_b parity remains exact
bs=2/4/5/6 still fail, unless already fixed by local changes
```

If bs=1 regresses, stop and repair the 11.11 owner before continuing.

### 2. Build A Rank-Local First-Owner Census

For each failing verify event, collect first owner by rank, not only rank0:

```text
batch size
verify event id
request id
rank
first mismatching layer/submodule
boundary before owner
boundary at owner
dtype
max_delta / mean_delta
normal value sample
target-verify value sample
```

Required cases:

```text
bs=1 targeted trace after wo_b fix
bs=2 targeted trace after wo_b fix
bs=1/2/4/5/6 matrix summary
```

Do not stop at visible token mismatches.  The useful output is a rank-local
hidden-state owner table.

### 3. Group Owners By Mechanism

Group the census into mechanism buckets:

```text
indexer FP8 quantization / indexer_query_fp8_values
MoE output / expert routing / expert reduction
attention q/k/v norm/RoPE local path
later-layer attention consumer
projection/reduce path
component/C4/C128 compressor state
oracle construction
```

For each bucket, answer:

- Does it appear in bs=1, bs=2, or both?
- Is it rank-local before any reduce?
- Is its input exact?
- Does it involve uint8/FP8 quantization?
- Does it involve MoE route or expert output?
- Does SGLang have a different target-verify handling for this component?
- Is this likely one common cause or an independent owner?

### 4. Source-Parity Check The Top Buckets

Do source parity before trying many local flags.

For `layer2.indexer_query_fp8_values`, compare mini and SGLang around:

```text
indexer query construction
FP8 activation quantization
scale selection
rounding/saturation behavior
per-row vs batched quantization
target-verify row ordering
component/indexer cache state used by target verify
```

For `layer0.moe_output`, compare:

```text
router input and topk indices
expert input values
expert weight/cache path
shared expert path
expert output reduction/aggregation
target-verify row shape effects
```

For later-layer attention drift, compare:

```text
q/k/v producer inputs
RoPE positions
KV read/write locations
target-verify decode-row attention path
whether upstream indexer/MoE drift already explains it
```

### 5. Try Minimal Mechanism Oracles

Acceptable temporary probes:

- force row-invariant FP8 quantization for indexer inputs;
- compare FP8 quantized uint8 values against a per-row reference;
- force the same indexer query path for normal decode and target verify under
  `sglang_prefill_extend`;
- trace MoE input, topk, expert outputs, shared expert output, and final
  aggregated output for one failing event;
- force row-invariant MoE or reference MoE for one layer/event as an oracle;
- compare only selected layers/ranks/rows under debug envs.

Not acceptable as final behavior:

- a bs-specific, rank-specific, or token-specific branch;
- disabling accepted commit;
- sequentially recomputing accepted rows as the final MTP path;
- silently changing global precision without a correctness and performance note.

### 6. Fix Or Choose The Next Focused Target

If one common owner explains most failures, fix it in this target and rerun:

```text
bs=1/2/4/5/6 exactness matrix
bs=7/8/16 light exposure only if the matrix passes
```

If multiple independent owners exist, do not patch them all loosely.  Write the
next target around the highest-priority owner, using this priority:

1. Owner that explains bs=1 and bs=2 drift together.
2. Owner that is earliest in layer order.
3. Owner whose input is exact and output is non-exact.
4. Owner with clear SGLang source behavior to port/adapt.
5. Owner with smallest correctness-only implementation surface.

## Success Criteria

Minimum:

```text
rank-local first-owner census exists for bs=1 and bs=2
owners are grouped by mechanism
top owner has source-parity evidence
accepted commit remains enabled
no new batch/rank/request/token special branch is introduced
```

Full:

```text
one common downstream owner is fixed
bs=1/2/4/5/6 exact
bs=7/8/16 exposure passes or has a new precise owner
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- bs=1 layer0 `wo_b` parity regresses;
- the only passing fix is bs/rank/request/token special casing;
- exactness requires disabling accepted commit;
- owners are multiple and independent enough that one target cannot safely fix
  them all;
- the top owner requires a larger SGLang component port.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_rank_local_downstream_parity_census/README.md
```

Include:

- reproduction of the 11.11 state;
- rank-local first-owner census table;
- mechanism grouping;
- SGLang source-parity notes for top owner buckets;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- next focused correctness target if needed, or TARGET 11.3 go/no-go.

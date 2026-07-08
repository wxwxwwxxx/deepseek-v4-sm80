# TARGET 11.15: DSV4 SM80 MTP MoE Output Sub-Boundary Parity

## Status

Next after TARGET 11.14.

TARGET 11.14 closed the q-path precision boundary for the required rank6
event0 rows:

```text
q_path_hidden_input: bit-exact
wq_a / q_lora: bit-exact
q_norm: bit-exact
wq_b / q_wqb_output: bit-exact after row-invariant target-verify local projection
q_norm_rope: bit-exact
```

Full eager MTP exactness is still not proven:

```text
bs=1 fail
bs=2 fail
bs=4 pass
bs=5 fail
bs=6 fail
```

The next owner exposed by the same operator framework is MoE output:

```text
bs=2 event0 rank6:
  layer0.moe_input exact
  layer0.moe_output drifts

bs=1 event0 rank6:
  layer7.moe_input exact
  layer7.moe_output drifts
```

This target should split `moe_input -> moe_output` into sub-boundaries and fix
or precisely no-go the first MoE owner.

## Goal

Make MTP target-verify MoE output equivalent to normal target decode for the
same visible row, or identify the exact MoE sub-boundary that needs a focused
follow-up.

The target passes when one of these is true:

1. The MoE owner is fixed and the `sglang_prefill_extend` exactness matrix
   improves or passes.
2. Or the MoE owner is classified into a precise sub-boundary with enough
   source-parity evidence for the next target to fix that sub-boundary.

Do not start indexer FP8, later-layer attention, CUDA graph, throughput tuning,
C128 boundary gates, or acceptance tuning in this target.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_q_wqb_q_lora_precision_boundary/README.md
performance_milestones/target11_mtp_operator_parity_q_norm_rope/README.md
performance_milestones/target11_mtp_rank_local_downstream_parity_census/README.md
prompts/TARGET_11.14_dsv4_sm80_mtp_q_wqb_q_lora_precision_boundary_parity.md
```

Important TARGET 11.14 result:

```text
q path is no longer the first owner for the required rows.
MoE input is exact; MoE output drifts.
```

Keep queued but do not fix yet:

```text
indexer FP8 query
later-layer attention
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
```

Relevant mini MoE path:

```text
router input / gate logits
topk ids and topk weights
routed expert input
routed expert output
shared expert input
shared expert output
expert aggregation / reduce-once
final moe_output
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
/workspace/sglang-main/python/sglang/srt/layers/moe/utils.py
```

Relevant SGLang behavior:

```text
DeepseekV2MoE
router/topk
routed experts
shared experts
optional reduce-scatter / all-reduce
skip-post-expert-all-reduce utilities
```

Use SGLang source behavior as the preferred contract when mini and SGLang differ
around target-verify MoE execution, expert aggregation, or communication.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id, or
  observed token numerical branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6` as the fix.
- Do not reopen target-verify metadata, attention/KV, `wo_a`, `wo_b`, q/RoPE,
  or q/wq_b unless new operator evidence proves those attributions were wrong.
- Do not fix indexer FP8 in this target.
- Do not rely only on visible token mismatches; use operator sub-boundary
  allclose evidence.

## Work Plan

### 1. Reproduce The MoE Owner

Use the same contract:

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
MINISGL_DSV4_MTP_OPERATOR_PARITY=1
```

Reproduce at least:

```text
bs=2 event0 rank6 layer0.moe_input exact -> layer0.moe_output drift
bs=1 event0 rank6 layer7.moe_input exact -> layer7.moe_output drift
```

If MoE no longer reproduces, rerun the operator census and follow the new first
owner.

### 2. Extend Operator Parity Harness For MoE

Reuse the 11.13/11.14 operator parity harness and add MoE records for:

```text
moe_input
router_logits
topk_ids
topk_weights
routed_expert_input
routed_expert_output
shared_expert_input
shared_expert_output
expert_aggregate_before_reduce
expert_reduce_output / post_all_reduce if applicable
moe_output
```

For each boundary, record:

```text
normal path/kernel
target-verify path/kernel
shape
dtype
allclose
bit-exact flag if cheap
max_delta / mean_delta
first differing index
sample values
owner verdict
```

For sparse/routed outputs, also record enough routing metadata to compare rows:

```text
expert ids
expert weights
token/expert counts if available
whether row order changed
whether padding rows were included
```

### 3. Source-Parity Table For MoE

Before trying many flags, write a source-parity table:

```text
concept
SGLang behavior
mini normal decode behavior
mini target-verify behavior
same/different/unknown
action
```

Cover:

- router input and dtype;
- gate logits and topk selection;
- topk weights normalization;
- expert routing/dispatch order;
- routed expert weight path: WNA16/Marlin/cached BF16/fallback;
- routed expert activation precision;
- shared expert input and output path;
- routed + shared aggregation order;
- all-reduce / reduce-once / reduce-scatter behavior;
- row-batched vs per-row behavior;
- active/padded target-verify rows and whether padded rows are excluded.

### 4. Micro Allclose / Oracles

Try these in order:

1. Router/topk oracle:
   - compare gate logits, topk ids, and topk weights for normal vs target verify.
2. Routed expert oracle:
   - with matching routing, compare routed expert input/output.
3. Shared expert oracle:
   - compare shared expert input/output.
4. Aggregation/reduce oracle:
   - compare pre-reduce aggregate and post-reduce output.
5. Row-invariant oracle:
   - if target verify uses multi-row expert batches while normal decode is
     single-row, run the target verify MoE sub-path row-by-row as a diagnostic.
6. SGLang-style oracle:
   - if SGLang's MoE target-verify semantics differ materially, port/adapt the
     relevant behavior under `sglang_prefill_extend`.

Do not promote a slow reference MoE path as the final runtime unless the report
explicitly marks it as a temporary correctness oracle and writes a follow-up
performance plan.

### 5. Fix Or Precisely No-Go

Preferred fixes:

1. If routing differs:
   - align router/topk dtype, shape, and row order.
2. If routed expert output differs:
   - make the expert backend row-invariant or align with SGLang/normal decode.
3. If shared expert output differs:
   - align shared expert precision/path and any staging/copy behavior.
4. If aggregation/reduce differs:
   - align reduce dtype/op/backend and aggregation order.
5. If padded rows contaminate MoE:
   - exclude padded target-verify rows from routing, expert execution, and
     aggregation.

Do not fix by branching on batch size, rank, request id, expert id, or token id.

### 6. Validate Incrementally

After a fix:

```text
MoE operator parity for bs=2 event0 rank6 layer0
MoE operator parity for bs=1 event0 rank6 layer7
bs=1 targeted trace
bs=2 targeted trace
bs=1/2/4/5/6 exactness matrix
```

If the matrix still fails, rerun enough of the operator census to identify the
next owner.  Expected queued owner is indexer FP8 unless MoE exposes another
sub-boundary.

## Success Criteria

Minimum:

```text
MoE output owner is reproduced
first MoE sub-boundary is identified with operator records
MoE source-parity table is written
accepted commit remains enabled
no batch/rank/request/token/expert special branch is introduced
```

Full:

```text
MoE sub-boundary fixed for known rows
bs=1/2/4/5/6 exact, or next owner is identified by the operator framework
indexer FP8 remains queued or becomes the next focused target with evidence
TARGET 11.3 remains blocked or unblocked with explicit evidence
```

## Stop Lines

Stop and report if:

- MoE owner cannot be reproduced and the operator census must be refreshed;
- exactness requires batch/rank/request/token/expert special casing;
- MoE routing differs because the input is not actually exact;
- fixing MoE requires a larger SGLang MoE backend port;
- MoE is fixed but independent indexer/later-attention owners remain.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
```

Include:

- reproduction of the 11.14 MoE owner;
- operator parity harness extensions for MoE;
- MoE source-parity table against SGLang;
- router/topk, routed expert, shared expert, aggregation/reduce probe results;
- implementation summary or precise no-go;
- exactness matrix for `sglang_prefill_extend`;
- accepted commit stats;
- next owner and next focused target if needed, or TARGET 11.3 go/no-go.

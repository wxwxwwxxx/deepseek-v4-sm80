# TARGET 11.17: DSV4 SM80 MTP MoE Pre-Reduce Drifting-Rank Parity

## Status

Next after TARGET 11.16.

TARGET 11.16 closed the post-reduce-only hypothesis as a no-go. The rank6
`expert_reduce_output` drift was real, but it was produced by TP SUM after other
ranks already had a local pre-reduce drift:

```text
bs=2 layer0:
  rank0 expert_aggregate_before_reduce already drifts
  ranks1-7 expert_aggregate_before_reduce are bit-exact
  all_reduce SUM propagates rank0 drift to every rank

bs=1 layer7:
  rank0 and rank7 expert_aggregate_before_reduce already drift
  ranks1-6 expert_aggregate_before_reduce are bit-exact
  all_reduce SUM propagates rank0/rank7 drift to every rank
```

Therefore the next owner is still inside MoE, but one boundary earlier and on
the drifting TP ranks, not on the clean rank6 collective output.

Current exactness matrix under `MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend`
is still:

```text
bs=1 fail
bs=2 fail
bs=4 pass
bs=5 fail
bs=6 fail
```

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Find and fix, or precisely no-go, the first local MoE sub-boundary that causes
`expert_aggregate_before_reduce` to drift on the TP ranks identified by TARGET
11.16.

The target passes when one of these is true:

1. The drifting-rank local MoE owner is fixed, `expert_aggregate_before_reduce`
   becomes bit-exact on the required ranks, and the `sglang_prefill_extend`
   exactness matrix improves or passes.
2. Or the target proves a precise remaining owner such as routed expert output,
   shared expert output, route weight scaling, FP32/BF16 aggregation order, or
   fused runner staging, with enough evidence for the next target to implement
   the fix.

Do not move to indexer FP8, later attention, CUDA graph, acceptance tuning, or
throughput profiling until this local MoE pre-reduce owner is closed.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_moe_post_reduce_parity/README.md
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
performance_milestones/target11_mtp_q_wqb_q_lora_precision_boundary/README.md
prompts/TARGET_11.16_dsv4_sm80_mtp_moe_post_reduce_parity.md
prompts/TARGET_11.15_dsv4_sm80_mtp_moe_output_subboundary_parity.md
```

Important TARGET 11.16 result:

```text
The all_reduce is not independently bad. It sums a local pre-reduce drift from
rank0/rank7 into every rank. Rank6 is a clean local contributor in the inspected
probes, so do not debug rank6 as the source.
```

Required starting probes:

```text
bs=2 event0 layer0 rank0:
  expert_aggregate_before_reduce non-exact, allclose
  max_delta = 0.000244140625

bs=1 event0 layer7 rank0:
  expert_aggregate_before_reduce non-exact, allclose
  max_delta = 0.00048828125

bs=1 event0 layer7 rank7:
  expert_aggregate_before_reduce non-exact, allclose
  max_delta = 0.000003814697265625
```

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
python/minisgl/distributed/impl.py
```

Relevant Mini MoE boundaries:

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
expert_reduce_output
moe_output
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v2.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
/workspace/sglang-main/python/sglang/srt/layers/moe/utils.py
```

Relevant SGLang behavior to inspect and cite:

```text
DeepseekV2MoE.forward_normal
MoEGate.forward and grouped/topk behavior for DSV4
fused expert runner output dtype and staging
shared expert staging
maybe_fuse_routed_scale_and_shared_add
routed scaling and shared-add order
hidden-dtype vs FP32 staging around local aggregate
```

Use SGLang source behavior as the preferred contract when Mini and SGLang
differ. If SGLang's exact fused backend is not available in Mini, still record
the contract and decide whether Mini should adapt it or keep a proven local
equivalent.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id,
  layer id, token id, expert id, or prompt-content special branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not fix indexer FP8 in this target.
- Do not change TP all-reduce, reduce-scatter, or skip-reduce policy unless the
  new pre-reduce evidence proves the local aggregate contract requires it.
- Do not reopen attention/KV, `wo_a`, `wo_b`, q/RoPE, or q/wq_b unless the new
  evidence proves earlier attribution was wrong.
- Do not optimize throughput or CUDA graph replay.
- Do not accept text similarity as success; use operator parity and exactness
  gates.

## Work Plan

### 1. Reproduce Drifting-Rank Local Aggregate Owners

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
MINISGL_DSV4_MTP_OPERATOR_PARITY=1
```

Reproduce at least:

```text
bs=2 event0 layer0 rank0:
  expert_aggregate_before_reduce drift

bs=1 event0 layer7 rank0:
  expert_aggregate_before_reduce drift

bs=1 event0 layer7 rank7:
  expert_aggregate_before_reduce tiny drift
```

If these no longer reproduce, rerun the rank-local operator census and follow
the new first owner.

### 2. Re-run Full MoE Sub-Boundary Census On Drifting Ranks

Do not restrict the first pass to rank6. For the drifting ranks above, record:

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
```

For each boundary, record:

```text
normal path/kernel
target-verify path/kernel
shape
dtype
stride
storage offset
contiguity
bit-exact flag
allclose flag
max_delta / mean_delta
first differing index
sample normal / target values
route ids and route weights for the visible row
expert histogram / plan if available
rank id and layer id
```

The first non-allclose or first output-significant allclose drift before
`expert_aggregate_before_reduce` becomes the next owner.

### 3. Split Aggregate Into Routed, Shared, Scale, And Add

If `routed_expert_output` and `shared_expert_output` are both exact but
`expert_aggregate_before_reduce` drifts, split the local combine more finely:

```text
routed output before route-weight scaling
route weights used for scaling
routed output after route-weight scaling
shared output before cast
shared output after cast
routed + shared local sum before any hidden-dtype cast
final pre-reduce aggregate
```

The key question is whether Mini differs from normal decode because of:

```text
route weight dtype or broadcast shape
FP32 vs BF16 cast order
batched target-verify row staging
shared expert output dtype
fused runner output dtype
in-place vs out-of-place aggregate
row selection / padding rows
```

If the drift is exactly at routed/shared add, compare against SGLang's
`maybe_fuse_routed_scale_and_shared_add` contract and decide whether Mini should
use hidden-dtype fused staging or preserve FP32 staging with a proven exact
row-invariant path.

### 4. Source-Parity Table

Write a source-parity table before promoting any fix:

```text
Concept
SGLang behavior
Mini normal decode
Mini target-verify
Verdict / action
```

Cover at least:

```text
router input dtype
router logits/topk dtype
route weight dtype
routed expert runner output dtype
shared expert output dtype
routed scaling dtype
routed/shared add dtype
final local aggregate dtype
padding-row treatment
target-verify multi-row treatment
```

If SGLang and Mini intentionally differ, write the rationale and prove the Mini
variant is exact for normal-vs-target required rows.

### 5. Try Minimal Correctness Fixes

Prefer small, source-aligned fixes:

```text
align route-weight scaling dtype/order
align shared expert cast order
align routed/shared aggregate order with SGLang
use a row-invariant target-verify local combine only if it solves the owner and
  does not branch on batch/rank/layer/token identity
add an opt-in diagnostic path before promoting any behavior change
```

Do not rewrite the entire MoE runner unless the source-parity table proves the
runner implements the wrong target-verify local aggregate contract.

### 6. Validation Gates

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

Operator parity gates:

```text
bs=2 event0 layer0 rank0 expert_aggregate_before_reduce
bs=1 event0 layer7 rank0 expert_aggregate_before_reduce
bs=1 event0 layer7 rank7 expert_aggregate_before_reduce
```

If fixed, also verify that the downstream collective no longer propagates this
drift:

```text
bs=2 event0 rank6 layer0 expert_reduce_output / moe_output
bs=1 event0 rank6 layer7 expert_reduce_output / moe_output
```

Exactness matrix:

```text
bs=1
bs=2
bs=4
bs=5
bs=6
```

Use the same six fixed prompts from TARGET 11.15/11.16 when `bs=6` is included.

If `expert_aggregate_before_reduce` becomes exact on the drifting ranks but the
exactness matrix still fails, run the rank-local operator census and name the
new first owner. Do not assume indexer FP8 without fresh evidence.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_moe_pre_reduce_drifting_rank_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
source-parity table against SGLang
drifting-rank MoE sub-boundary census
aggregate split evidence, if needed
before/after operator parity tables
exactness matrix
accepted commit stats
remaining owner or promotion/no-go verdict
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- The drifting rank's `moe_input` is already non-exact, because the owner moved
  before MoE and this target's premise is stale.
- The first owner is a SGLang-vs-Mini fused MoE staging contract that requires a
  larger runner replacement; document the exact contract and next step instead
  of applying a rank-special-case patch.
- A proposed fix only passes one batch size by branching on batch/rank/layer or
  visible token identity.
- The fix improves text smoke but not operator parity.
- The pre-reduce local aggregate becomes exact but the exactness matrix still
  fails; close this target with the new first owner rather than doing a broad
  MTP rewrite.

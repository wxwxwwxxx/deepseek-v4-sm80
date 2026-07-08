# TARGET 11.16: DSV4 SM80 MTP MoE Post-Reduce Parity

## Status

Next after TARGET 11.15.

TARGET 11.15 reproduced the MoE output owner and narrowed it to the
post-experts communication/cast boundary:

```text
expert_aggregate_before_reduce: bit-exact
expert_reduce_output: near-exact drift
moe_output: BF16-cast non-allclose
```

The report explicitly ruled out the visible-row router/topk path, routed
experts, shared experts, and pre-reduce aggregation as the output-significant
MoE owner for the required probes.

Current exactness matrix under `MINISGL_DSV4_TARGET_VERIFY_RUNTIME=sglang_prefill_extend`
is still not good enough:

```text
bs=1 fail
bs=2 fail
bs=4 pass
bs=5 fail
bs=6 fail
```

TARGET 11.3 graph/perf promotion remains no-go.

## Goal

Make the MoE post-experts reduce boundary equivalent between normal target
decode and MTP target-verify for the same visible row, or prove the exact
SGLang communication contract that Mini must port.

The target passes when one of these is true:

1. `expert_reduce_output` becomes bit-exact for the required normal-vs-target
   rows, and the `sglang_prefill_extend` exactness matrix improves or passes.
2. Or the target produces a precise no-go that identifies the exact reduce
   staging, dtype, cast, all-reduce, reduce-scatter, or skip-reduce contract
   difference, with enough SGLang source parity evidence for the next target to
   implement it.

Do not move to indexer FP8, later attention, CUDA graph, acceptance tuning, or
throughput profiling until this post-reduce owner is closed.

## Starting Evidence

Read first:

```text
performance_milestones/target11_mtp_moe_output_subboundary_parity/README.md
performance_milestones/target11_mtp_q_wqb_q_lora_precision_boundary/README.md
performance_milestones/target11_mtp_rank_local_downstream_parity_census/README.md
prompts/TARGET_11.15_dsv4_sm80_mtp_moe_output_subboundary_parity.md
```

Important TARGET 11.15 result:

```text
bs=2 event0 rank6 layer0:
  expert_aggregate_before_reduce exact
  expert_reduce_output max_delta = 0.000244140625, allclose but not bit-exact
  moe_output max_delta = 0.001953125, non-allclose

bs=1 event0 rank6 layer7:
  expert_aggregate_before_reduce exact
  expert_reduce_output max_delta = 0.00048828125, allclose but not bit-exact
  moe_output max_delta = 0.00390625, non-allclose
```

The temporary row-by-row post-reduce diagnostic in TARGET 11.15 did not close
bs=2/rank6/layer0, so do not assume the issue is only "batched all-reduce vs
single-row all-reduce" without further evidence.

## References

Mini:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/distributed/impl.py
python/minisgl/utils/dsv4_mtp_debug.py
python/minisgl/engine/engine.py
```

Known Mini MoE reduce locations:

```text
_dsv4_moe_reduce_once_input
dsv4.v1_moe_reduce_once_all_reduce
dsv4.routed_expert_all_reduce
dsv4.shared_expert_all_reduce
mini.moe.runner.post_all_reduce
mini.moe.post_all_reduce
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/models/deepseek_v2.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
/workspace/sglang-main/python/sglang/srt/layers/moe/utils.py
/workspace/sglang-main/python/sglang/srt/layers/moe/
```

Relevant SGLang behavior to inspect and cite:

```text
DeepseekV2MoE.forward
DeepseekV4 layer/expert dispatch
maybe_fuse_routed_scale_and_shared_add
should_skip_post_experts_all_reduce
reduce_scatter / all_reduce / skip-all-reduce choices
dtype and cast order around routed + shared expert aggregation
target-verify / extend-mode communication behavior if it differs from decode
```

Use SGLang source behavior as the preferred contract when Mini and SGLang
differ. If SGLang does not provide an equivalent path for this exact Mini mode,
state that clearly and prove the Mini-local contract instead.

## Non-Goals

- Do not add parent batch size, active verify length, request slot, rank id,
  layer id, token id, expert id, or prompt-content special branches.
- Do not disable accepted commit.
- Do not switch back to `legacy_target11_6`.
- Do not fix indexer FP8 in this target.
- Do not reopen attention/KV, `wo_a`, `wo_b`, q/RoPE, or q/wq_b unless the new
  reduce evidence proves TARGET 11.10-11.14 attribution was wrong.
- Do not optimize throughput or CUDA graph replay.
- Do not accept "text looks okay" as success; use operator parity and exactness
  gates.

## Work Plan

### 1. Reproduce The Post-Reduce Owner

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
bs=2 event0 rank6 layer0:
  expert_aggregate_before_reduce exact
  expert_reduce_output drift
  moe_output drift

bs=1 event0 rank6 layer7:
  expert_aggregate_before_reduce exact
  expert_reduce_output drift
  moe_output drift
```

If the owner no longer reproduces, rerun the operator census and follow the new
first owner. Do not keep debugging stale traces.

### 2. Build A Reduce Boundary Census

Extend the existing MTP operator parity/debug records around the post-experts
reduce path. For both normal decode and target-verify, record:

```text
pre_reduce tensor shape/dtype/stride/storage_offset/contiguous
pre_reduce visible-row sample and checksum
local rank contribution before communication
communication op label
communication backend: torch distributed / PyNCCL / no-op
all_reduce vs reduce_scatter vs skip-reduce, if applicable
communication input dtype
communication output dtype
post_reduce tensor shape/dtype/stride/storage_offset/contiguous
post_reduce visible-row sample and checksum
final cast dtype and cast site
moe_output visible-row sample and checksum
```

For the required rows, also record:

```text
max_delta / mean_delta
first differing index
normal value
target-verify value
rank-local contribution values across all ranks if cheap
whether the BF16 cast changes the allclose verdict
```

The key question is whether the drift appears:

```text
before communication
during communication
after communication but before final cast
during final hidden-dtype cast
because Mini performs a reduce that SGLang would skip or absorb elsewhere
because Mini reduces with a different dtype/order/shape than SGLang
```

### 3. Compare Against SGLang's Communication Contract

Write a source-parity table before trying many fixes:

```text
Concept
SGLang behavior
Mini normal decode
Mini target-verify
Verdict / action
```

Cover at least:

```text
routed expert output dtype
shared expert output dtype
routed/shared aggregation dtype
post-experts all-reduce condition
skip-post-experts-all-reduce condition
reduce-scatter condition
whether downstream projection/parallelism absorbs a SUM
final cast order
target-verify/extend-mode difference, if any
```

If SGLang uses reduce-scatter or skip-all-reduce for this mode, do not emulate
that by a cosmetic flag. Identify the downstream owner that consumes the partial
or reduced tensor and port the actual contract, or close with a precise no-go.

### 4. Try Minimal Correctness Fixes

Prefer small, source-aligned fixes:

```text
align final cast order with SGLang
align pre-reduce dtype/staging with SGLang
align skip-all-reduce or reduce-scatter condition with SGLang
reuse an existing Mini row-invariant reduce helper only if it is proven to solve
  the owner and does not introduce batch/layer special casing
```

Avoid broad rewrites of the MoE runner unless the source-parity table proves the
current runner implements the wrong communication contract.

If a diagnostic flag is useful, keep it opt-in and clearly named. Do not promote
it by default in this target unless the exactness matrix and smoke gates pass.

### 5. Validation Gates

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
bs=2 event0 rank6 layer0 expert_reduce_output
bs=1 event0 rank6 layer7 expert_reduce_output
```

Exactness matrix:

```text
bs=1
bs=2
bs=4
bs=5
bs=6
```

Use the same six fixed prompts from TARGET 11.15 when `bs=6` is included.

If `expert_reduce_output` becomes exact but the exactness matrix still fails,
run the rank-local operator census and name the next first owner. The queued
candidate from earlier work is indexer FP8, but do not assume it without fresh
evidence.

## Deliverables

Write:

```text
performance_milestones/target11_mtp_moe_post_reduce_parity/README.md
```

The README must include:

```text
summary verdict
implementation summary
source-parity table against SGLang
reduce boundary census
before/after operator parity tables
exactness matrix
accepted commit stats
remaining owner or promotion/no-go verdict
next recommended target
```

## Stop Lines

Stop and write a precise no-go if:

- `expert_aggregate_before_reduce` is no longer exact, because the owner moved
  upstream and this target's premise is stale.
- The reduce drift is caused by a SGLang contract Mini does not yet implement,
  such as skip-post-experts-all-reduce or reduce-scatter absorption, and the
  implementation would require a larger downstream graph ownership change.
- A proposed fix only passes one batch size by branching on batch/rank/layer or
  visible token identity.
- The fix improves text smoke but not operator parity.
- The exactness matrix still fails after `expert_reduce_output` is exact; in
  that case close this target with the new first owner instead of doing a broad
  MTP rewrite.

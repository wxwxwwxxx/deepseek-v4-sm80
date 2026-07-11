# TARGET 12.6025: DSV4 SM80 MoE Padding Live-Route Contract Fix

## Status

Current after TARGET 12.602.

TARGET 12.602 classified upward-padding drift as:

```text
BLOCKING_DUMMY_ROW_OR_STATE_CONTAMINATION
```

The first credible producer is the layer-0 MoE route-plan/grouped-expert
boundary. Live inputs, attention output, MoE input, route weights, and route
indices match before planning, but valid dummy routes alter global sorting,
expert counts/offsets, `num_tokens_post_padded`, live MoE output, and finally
sampled tokens.

## Purpose

Establish and implement one graph-safe MoE live-route contract:

> CUDA graph padding may provide static tensor capacity, but rows at or beyond
> `num_token_non_padded` must not participate in routing, expert histograms,
> sorting, expert block counts/offsets, grouped expert work, reductions, or live
> outputs.

Fix the owner once at the route-plan/backend boundary. Do not patch downstream
hidden states, logits, sampler output, or cache state.

## Required References

Mini evidence and implementation:

```text
performance_milestones/target12_cuda_graph_padding_live_row_classification/README.md
performance_milestones/target12_cuda_graph_padding_live_row_classification/raw/
python/minisgl/engine/graph.py
python/minisgl/core.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/marlin_wna16.py
python/minisgl/moe/fused.py
```

SGLang behavior to align with:

```text
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/moe.py
/workspace/sglang-main/python/sglang/srt/layers/moe/topk.py
/workspace/sglang-main/python/sglang/srt/layers/moe/hash_topk.py
/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py
```

SGLang carries `num_token_non_padded` as graph-visible state, masks padded
top-k IDs, and zeros padded top-k weights. Mini already has a partial precedent
in `python/minisgl/moe/fused.py`. Prefer adapting this contract.

Also inspect the vLLM Marlin assignment/align API for buffer and sentinel
semantics:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
```

## Required Work

### 1. Add One Stable Live-Row Input

Carry the actual live decode row count through graph staging as a one-element
device integer tensor with a stable address. During capture its capacity is the
captured M; during replay its value is the actual batch size.

- Do not read it back to the CPU in the hot path.
- Do not specialize or recapture per actual M.
- Exact-M and eager paths must use the same semantic contract.
- Keep `batch.size`, `batch.padded_size`, and `num_token_non_padded` separately
  observable in diagnostics.

### 2. Mask Routes At The Earliest Shared Boundary

Before any route histogram or alignment consumes top-k output:

```text
row < num_token_non_padded:
  preserve topk_ids and topk_weights exactly

row >= num_token_non_padded:
  topk_ids     = invalid route sentinel (normally -1)
  topk_weights = 0
```

Prefer folding this into an existing DSV4 top-k/hash-top-k or route-plan kernel.
One small captured kernel is acceptable as a first correct implementation, but
record its cost and avoid one new host launch per layer outside the graph.

The sentinel must agree with mini/Marlin alignment semantics and must never be
interpreted as a real expert.

### 3. Make The Route Plan Authoritative

Audit the runner/backend boundary. The runner currently constructs a
`DSV4MoEExecutionPlan`, while the Marlin WNA16 prepacked path can ignore it and
rebuild a plan from the complete padded `indices` tensor.

Refactor so the production Marlin path consumes one authoritative masked plan,
or prove that its internal rebuild consumes the already-masked IDs and produces
the same plan. Do not retain two semantically different plans.

For live routes, require identical:

```text
route IDs and weights
expert histogram
sorted route IDs
expert IDs and block offsets
num_tokens_post_padded
grouped expert inputs/outputs
```

Keep route/output/workspace allocations at captured maximum capacity while the
device-side work count remains dynamic.

### 4. Define Padded Output Semantics

Excluded routes may leave backend route slots unwritten. Ensure padded rows
cannot consume uninitialized `w13_out`, activation, `route_out`, or grouped
output storage in later MoE layers.

Use bounded zeroing or masked finalization where required. Do not zero the full
maximum workspace unnecessarily if a live/padded-row kernel can establish the
contract more cheaply.

Dummy rows must remain safe through residual/shared-expert execution even
though only live outputs are returned.

### 5. Preserve Static Graph And Communication Contracts

- CUDA graph addresses and tensor capacities remain static.
- TP ranks receive the same live-row scalar and route exclusion semantics.
- Collective count, shape, dtype, and ordering remain graph-stable.
- Dummy rows contribute zero semantic value to routed/shared reductions.
- No dynamic allocation or CPU synchronization enters decode replay.
- Exact-M behavior and current max16 small-batch latency must not regress
  materially.

### 6. Focused Correctness Gates

Reuse the TARGET 12.602 harness and actual generated candidate buckets:

```text
17 -> 24
33 -> 40
57 -> 64
64 -> 64 negative control
```

For at least two valid dummy poison profiles and two fresh processes, require:

1. live route plans are identical across poison profiles;
2. live layer-0 MoE output is bit-identical, or exactly explain and gate any
   residual backend-only numerical difference after route plans match;
3. live selected boundaries, logits, sampled tokens, and persistent cache state
   are poison-invariant;
4. no NaN/Inf, invalid route/location, stale storage, or lifecycle error;
5. same-shape repeat stability;
6. Chinese, English, code, arithmetic, and instruction text sanity passes.

After poison invariance passes, run the deferred exact-M versus padded-M logit
margin census. Pure shape-dependent BF16 drift may be accepted under the
TARGET 12.602 numerical contract; batch-shape token invariance is not required.

### 7. Performance Gate

Measure route-plan and MoE GPU time before/after at exact 16/64 and padded
17->24, 33->40, 57->64.

- No new eager decode fallback.
- No extra uncaptured per-layer host work.
- Current bs4 release baseline regression must be <= 2% and within repeat
  noise.
- Report whether excluding dummy expert blocks improves padded execution.
- Optimize only an overhead introduced by this fix; broad MoE tuning belongs
  to TARGET 12.61.

## Cleanup

Retain compact regression hooks/tests. Remove or disable-by-default broad
boundary dumps and large instrumentation added only for TARGET 12.602 once the
same evidence can be reproduced by the focused harness. Do not remove useful
debug harness artifacts from the milestone directory.

## Required Decision

```text
MOE_LIVE_ROUTE_CONTRACT_FIXED
BLOCKED_BY_MARLIN_DYNAMIC_ROUTE_PLAN
BLOCKED_BY_GRAPH_VISIBLE_LIVE_COUNT
BLOCKED_BY_RESIDUAL_POISON_OWNER
```

Only `MOE_LIVE_ROUTE_CONTRACT_FIXED` permits upward-padding promotion work to
continue.

## Non-Goals

- Do not implement graph-memory reservation; TARGET 12.603 owns it.
- Do not promote max64/max128 buckets; TARGET 12.605 owns that decision.
- Do not change precision, MTP, attention/cache ownership, page/chunk size, or
  M>512 behavior.
- Do not require exact-M and padded-M token bit-exactness after dummy poison is
  proven inert.
- Do not redesign the complete MoE backend or communication stack.

## Stop Conditions

Stop after poison invariance and performance gates pass, or after the first
remaining blocker is proven to require a separate backend/kernel change. Do
not continue polishing unrelated MoE kernels.

## Output

```text
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
```

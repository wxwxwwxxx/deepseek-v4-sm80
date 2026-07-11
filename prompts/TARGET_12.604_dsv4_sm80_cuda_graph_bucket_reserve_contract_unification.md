# TARGET 12.604: DSV4 SM80 CUDA Graph Bucket And Reserve Contract Unification

## Status

Current after TARGET 12.603.

TARGET 12.603 installed a safe DSV4/sm80 conservative graph-memory estimator
before KV planning. It validated explicit max16/max64/max128 bucket lists, but
the runtime still has two policy surfaces:

```text
Engine graph-memory estimator reads config.cuda_graph_bs
GraphRunner can independently derive buckets from cuda_graph_max_bs
```

Before generated buckets become a release default, these paths must consume one
resolved policy. Otherwise the planner can reserve one graph set while the
runner captures another.

## Purpose

Implement one pure, observable CUDA graph bucket resolver that runs before KV
planning and becomes authoritative for:

```text
graph-memory estimate
KV reserve ledger
GraphRunner capture list/order
padding dispatch
benchmark/run configuration
release diagnostics
```

This is a contract-unification target. Preserve the current no-env max16
default and do not select or promote max64/max128 here.

## Required References

```text
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
performance_milestones/target12_cuda_graph_memory_reserve_planner/README.md
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
python/minisgl/engine/config.py
python/minisgl/engine/engine.py
python/minisgl/engine/graph.py
python/minisgl/engine/graph_memory.py
benchmark/offline/deepseek_v4_perf_matrix.py
/workspace/vllm-dsv4-docker/vllm/config/vllm.py
/workspace/sglang-main/python/sglang/srt/server_args.py
```

Use the TARGET 12.60 vLLM/SGLang source census. Do not introduce another
independent generator.

## Required Work

### 1. Define One Resolved Policy Object

Create a pure resolver whose result records at least:

```text
enabled/disabled
source mode: release_default | explicit_list | explicit_max | disabled
requested list/max
effective max_running_req
resolved sorted unique bucket tuple
resolved maximum
generation rule/version
validation/cap reason
```

The result must not allocate CUDA memory or depend on post-KV free memory. It
must be available before graph-memory estimation and final KV page selection.

### 2. Define Precedence And Validation

Use an explicit contract:

1. `cuda_graph_bs=[]`, `cuda_graph_max_bs=0`, or an explicit graph-disable
   switch disables capture and produces an empty policy/reserve.
2. A non-empty explicit bucket list is authoritative after positive,
   sorted/unique validation. If an explicit maximum is also supplied and
   conflicts, fail clearly rather than silently choosing one.
3. A max-only request generates buckets with the TARGET 12.60 aligned rule:

```text
[1,2,4]
step 8 through 256
step 16 from 272 through 512
always include the exact requested endpoint
```

4. Omitted policy on the current DSV4 A100/sm80 release path remains exactly
   `[1,2,4,8,16]` until TARGET 12.605 promotes another default.
5. Automatic policy must not exceed effective `max_running_req`. Explicit
   incompatible requests fail clearly; do not silently claim the larger list.
6. The release generator accepts practical maxima only through 512. Existing
   isolated 1024/2048 smoke tooling remains explicit diagnostic behavior and
   must not leak into the release resolver.

Preserve non-DSV4 and non-sm80 behavior unless a shared helper can do so without
changing existing semantics.

### 3. Resolve Before Every Consumer

Run resolution during configuration normalization, before:

```text
graph-memory estimation
request/KV capacity planning
GraphRunner construction
```

Pass the exact resolved tuple to both
`estimate_dsv4_sm80_graph_memory` and `GraphRunner`. Remove or bypass the
independent GraphRunner regeneration for an already-resolved policy.

Add a fail-fast invariant after capture:

```text
estimated/resolved graph_bs == GraphRunner requested/captured graph_bs
```

Capture failure may still be handled by the existing fail-closed policy, but a
configuration mismatch is a programming error and must not be hidden as a
normal capture failure.

### 4. Keep Planner Accounting Coherent

The TARGET 12.603 estimator must use the resolved list, count, and maximum.
Reports must carry the resolver mode and exact tuple alongside:

```text
estimate + margin
actual graph bytes
lost KV pages/tokens
effective sequence width
post-capture free memory
```

Disabled graphs must reserve zero graph bytes and zero graph safety margin.
Explicit `num_pages` behavior remains unchanged: preserve a safe override and
fail an unsafe one without silently shrinking it.

Do not recalibrate estimator constants or reduce the 512 MiB safety margin in
this target. TARGET 12.605 will judge policy utility using the safe ledger.

### 5. Focused Tests

Use CPU/unit/no-weight tests first. Cover at least:

```text
omitted DSV4 release policy -> current [1,2,4,8,16]
explicit list, including unsorted/duplicate normalization or rejection contract
max-only 16, 64, and 128
non-step endpoint such as 67 is included exactly
disabled [], max=0, and explicit disable
conflicting explicit list and max
max above effective max_running_req
max above release limit 512
non-DSV4/non-sm80 preservation
estimator tuple equals resolved tuple
```

Then run one fresh TP8 repaired-production smoke, preferably max64, to prove:

- planner and GraphRunner report the same tuple;
- estimate/actual comparison and KV ledger still pass;
- one exact and one padded decode replay work;
- repaired MoE live-route contract remains healthy;
- no eager fallback occurs for covered M;
- no-env max16 behavior remains unchanged in a focused regression.

Do not rerun the full max16/64/128 matrix already completed by TARGET 12.603.

## Acceptance Gates

- exactly one resolver/generator owns release bucket semantics;
- planner and runner cannot observe different lists;
- max-only configuration really generates the requested policy;
- current omitted no-env default remains max16;
- disabled mode reserves/captures nothing;
- explicit-list/max conflicts and request-capacity violations fail clearly;
- no new CUDA allocation, synchronization, or hot-path work;
- focused TP8 correctness, planner, and memory gates pass;
- configuration and report telemetry identify the resolved policy unambiguously.

## Non-Goals

- Do not promote max64/max128 or choose the release winner.
- Do not tune graph estimator constants or safety margin.
- Do not port temporary full-model graph profiling.
- Do not change MoE, attention/cache, precision, MTP, page/chunk size, or
  M>512 behavior.
- Do not run a broad serving performance matrix.

## Required Decision

```text
BUCKET_RESERVE_CONTRACT_UNIFIED_READY_FOR_12.605
BLOCKED_BY_CONFIG_PRECEDENCE
BLOCKED_BY_PLANNER_RUNNER_POLICY_MISMATCH
BLOCKED_BY_RELEASE_DEFAULT_REGRESSION
```

## Output

```text
performance_milestones/target12_cuda_graph_bucket_reserve_contract_unification/README.md
```

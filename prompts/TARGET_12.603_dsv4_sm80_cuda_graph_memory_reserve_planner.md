# TARGET 12.603: DSV4 SM80 CUDA Graph Memory Reserve Planner

## Status

Current after TARGET 12.6025 closed the blocking MoE live-route contract with
`MOE_LIVE_ROUTE_CONTRACT_FIXED`.

Use the repaired release graph as the memory baseline. Its graph contains the
stable live-row scalar, captured route mask, authoritative Marlin route plan,
and padded-output finalize kernel; an estimate from the pre-fix graph is stale.

## Purpose

Adapt vLLM's CUDA graph memory accounting principle into mini: estimate or
profile graph-pool memory before final KV capacity is selected, subtract that
budget from KV-available memory, then validate the estimate against the actual
capture.

The result must be an automatic release mechanism, not another environment
recipe users need to learn.

This target must produce trustworthy memory planning, not select max64 versus
max128. TARGET 12.605 owns that promotion decision.

## Source Oracle

Read and cite the exact behavior in:

```text
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_worker.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
/workspace/vllm-dsv4-docker/vllm/config/cache.py
/workspace/sglang-main/python/sglang/srt/server_args.py
python/minisgl/engine/engine.py
python/minisgl/engine/graph.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/models/deepseek_v4.py
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
performance_milestones/target08_marlin_wna16_old_address_root_cause/README.md
performance_milestones/target08_marlin_wna16_safe_release_arena_capacity/README.md
performance_milestones/target08_marlin_wna16_release_component_clear_promotion/README.md
```

vLLM's relevant pattern is: profile model/activation and CUDA graph memory,
subtract non-KV memory from the requested device budget, plan KV, then compare
actual graph-pool memory with the estimate. Reuse this lifecycle rather than
inventing an unrelated fixed `memory_ratio` tweak.

In this source tree, vLLM initializes a minimal profiling KV cache, captures
representative graph descriptors in a temporary pool, estimates shared
first-capture plus per-graph cost, destroys profiling graphs/state, and then
uses the estimate in available-KV calculation. Cite exact source lines and
state what mini adapts versus what cannot be copied directly.

## Required Work

### 1. Define The Memory Contract

Separate and report:

```text
weights and persistent transformed-weight caches
fixed SWA/component/request-table state
peak activation/workspace allowance
estimated CUDA graph pool bytes
graph safety margin
variable KV bytes/page
final KV pages/tokens
actual CUDA graph pool bytes
post-capture and post-workload physical free bytes
```

Request capacity, model/effective max sequence length, graph maximum, bucket
count, and KV token capacity must remain distinct dimensions.

Avoid double counting peak activation that becomes part of the graph private
pool. Report allocator-live/reserved bytes and physical `mem_get_info` deltas,
but use synchronized physical bytes as the authoritative capacity ledger.

### 2. Audit The Profiling Lifecycle Before Full Weights

Build a no-weight/partial-model lifecycle harness first. Prove the intended
sequence can create and destroy:

```text
temporary minimal KV/component state
temporary request/page tables and attention metadata
temporary CUDA graph execs and graph pool
graph-visible MoE live-row/mask/finalize inputs
```

The cleanup order must be explicit:

1. synchronize profiling work;
2. destroy every graph exec and wrapper reference;
3. detach model/backend/global-context references to temporary KV and metadata;
4. release the temporary graph pool;
5. release temporary KV/component/page-table storage;
6. collect/empty allocator caches only where required;
7. verify no tensor, graph, backend, or global context retains an old storage
   pointer before formal KV allocation.

Use storage-range/sentinel probes where useful. The previous Marlin release and
component-arena milestones showed that allocator-visible free memory alone does
not prove old-address safety. Do not proceed to a full model until this harness
passes without stale-pointer, old-storage, or illegal-access behavior.

Do not rebuild or release transformed Marlin weights during graph profiling.
The production prebuild/release timing and capacity credit remain unchanged.

### 3. Implement A Pre-KV Graph Estimate

Prefer a vLLM-like profiling path:

1. initialize the model and graph-relevant backends;
2. create only the minimal valid temporary KV/component state needed by the
   profile;
3. profile representative graph descriptors in a temporary graph pool;
4. destroy temporary graphs/state and verify no stale storage survives;
5. synchronize the conservative estimate across TP ranks;
6. subtract estimate plus safety margin before final KV page planning;
7. allocate final KV/page tables and perform the real capture;
8. compare actual and estimated bytes.

Profile the same bucket generator, capture order, graph-pool sharing, greedy
sampler mode, DSV4 metadata width, and repaired MoE graph used by production.
Do not estimate a simplified graph and apply it to the release graph.

Adapt vLLM's estimator structure where evidence supports it:

```text
shared_pool_bytes ~= representative first/largest capture
per_graph_bytes   ~= representative incremental capture(s)
total_estimate    = shared_pool_bytes + conservative remaining-graph estimate
```

Mini captures largest to smallest in a shared pool. Verify whether one
first-capture plus a constant per-graph term is conservative across max16/64/128
before using that formula. If graph modes/pools overlap, take the maximum shared
owner rather than summing mutually exclusive pools.

If a temporary capture cannot be made lifecycle-safe, use a conservative,
measured estimator derived from bucket maximum/count and audited owners. Clearly
label it as an estimator and keep the API replaceable by future profiling. Do
not hide a hard-coded A100 number inside generic KV arithmetic.

Do not spend unlimited time forcing temporary full-model capture. After three
focused lifecycle attempts fail for the same reason, switch to the conservative
measured estimator and record the exact missing primitive.

Use configured/model max sequence width for conservative metadata estimation;
do not make the estimate circular by silently shrinking max sequence after KV
planning.

Explicitly resolve the planning dependency:

```text
KV pages -> KV token capacity -> effective max sequence
effective max sequence -> request/page-table bytes and DSV4 graph metadata width
those fixed/graph bytes -> memory left for KV pages
```

Use either a conservative requested/model-width upper bound or a monotonic
fixed-point planner with a provable safe result. If using fixed point, report
every term at convergence and assert that recomputing allocations from the
selected pages cannot exceed the budget. Do not perform trial GPU allocations
and catch OOM as the planning algorithm.

### 4. Integrate The Planner

Before final KV allocation, compute:

```text
requested_device_budget
- weights/transformed-weight persistent bytes
- fixed SWA/component bytes
- request/page-table bytes at the planned effective sequence width
- non-graph activation allowance not already counted in graph pool
- graph estimate at the same or a conservative larger metadata width
- graph safety margin
= variable KV budget
```

Convert graph estimate plus margin to exact lost pages/tokens using the DSV4
variable cache bytes per page. Synchronize the maximum required reserve across
TP ranks.

Start with the TARGET 12.60 conservative 512 MiB DSV4/A100 safety margin unless
fresh estimate-error evidence justifies another value. Keep model-specific
calibration outside generic cache arithmetic and report it visibly.

After formal capture, compare actual physical graph bytes with the estimate.
Record absolute/relative error and remaining safety margin. Do not allocate a
second persistent KV cache or recapture during serving.

### 5. Failure And Override Behavior

- Default behavior chooses a graph budget automatically.
- Explicit `num_pages` or KV-memory override remains authoritative but reports
  whether graph headroom is unsafe.
- Under-estimation must fail closed, lower/disable graph buckets, or produce a
  clear initialization error; it must not become a later CUDA OOM.
- Do not silently reduce `max_running_req` or change the requested graph policy.
- No allocation or profiling work may enter the decode hot path.
- A profile/estimate failure must leave a clean state before fallback planning;
  never continue after a partially destroyed graph/KV profile.
- If actual capture exceeds estimate plus margin, fail initialization or
  disable/lower graphs in a clearly reported way. Do not attempt unsafe KV
  reallocation inside the same initialized CUDA engine.

### 6. Validation Matrix

Start with no-weight/partial-model lifecycle tests, then use fresh TP8 full
processes for finalists:

```text
current max16
candidate max64
candidate max128
```

For each, compare predicted and actual graph bytes, final pages/tokens,
post-capture free memory, short/medium workload free memory, and initialization
time. Test an intentionally too-small reserve and an explicit KV override to
prove failure behavior.

Use the repaired TARGET 12.6025 production path in every full-model row. At
least one padded boundary per candidate must confirm graph replay and MoE live
route correctness remain intact; do not rerun the entire poison investigation.

Run two fresh TP8 processes for each final planner mode. Keep scenario-sized
request capacity diagnostic-only; the main ledger uses ordinary serving
defaults. Preserve the legal single-request 1M smoke separately rather than
requiring 1M and high concurrency together.

Acceptance targets:

- no stale pointer, old-storage, or CUDA lifecycle error;
- prediction never underestimates beyond the safety margin in tested cases;
- post-capture free >= 2 GiB and post-workload free >= 1 GiB;
- capacity loss is explained exactly in pages/tokens;
- repeated fresh-process planning is stable;
- current max16 latency/throughput does not materially regress;
- estimate overhead is reported separately from engine load and capture time;
- no-env `LLM(model_path)` receives the safe automatic plan.

Do not decide between max64 and max128 here. Produce trustworthy memory inputs
for TARGET 12.605.

## Non-Goals

- Do not promote generated buckets.
- Do not fix padding numerical behavior.
- Do not reopen the fixed MoE dummy-route investigation unless a regression gate
  fails.
- Do not require simultaneous high concurrency and 1M context.
- Do not tune kernels or M>512.
- Do not retain a second persistent KV/cache copy after profiling.

## Required Decision

```text
GRAPH_RESERVE_PLANNER_READY_FOR_12.605
GRAPH_PROFILE_UNSAFE_USE_CONSERVATIVE_ESTIMATOR
BLOCKED_BY_TEMPORARY_GRAPH_STORAGE_LIFECYCLE
BLOCKED_BY_ESTIMATE_ERROR_OR_CAPACITY
```

The report must give TARGET 12.605 exact planner inputs for max16/64/128:
estimate, margin, actual bytes, error, lost pages/tokens, remaining capacity,
and post-workload headroom.

## Output

```text
performance_milestones/target12_cuda_graph_memory_reserve_planner/README.md
```

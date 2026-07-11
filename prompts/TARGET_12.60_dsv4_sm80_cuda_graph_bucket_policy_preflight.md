# TARGET 12.60: DSV4 SM80 CUDA Graph Bucket Policy Preflight

## Status

Ready after TARGET 12.597.

TARGET 12.597 proved the legal model-default 1M total-sequence gate:

```text
prompt_len = 1048568
decode_len = 8
total      = 1048576
```

It also exposed a configuration ambiguity that this target must resolve before
large CUDA graph buckets are integrated: the offline benchmark can silently
set `max_running_req` from the scenario batch size, while ordinary
`LLM(model_path)` uses the serving default. These modes have different static
request-table and graph-memory costs.

## Scope Decision

The practical optimization range is decode batch / graph row count `M <= 512`.

- Measure and design a useful bucket policy through `M=512` while capacity and
  performance remain practical on A100 80 GB.
- `M=1024` and `M=2048` are isolated feasibility smoke points only. Do not
  build a dense ladder around them and do not tune kernels specifically for
  them in this target.
- Do not require 1M context and high concurrency to coexist at useful
  throughput. Extreme combinations need only configuration, admission, and
  basic correctness smoke coverage.
- Continue to treat 1M single-request support and practical short/medium
  context serving concurrency as separate capabilities.

## Purpose

Produce an evidence-backed, vLLM/SGLang-aligned CUDA graph bucket contract for
TARGET 12.605. Determine:

1. how buckets should be generated and padded;
2. the largest useful release bucket at or below 512;
3. the graph-pool, static-buffer, and KV-capacity cost of each candidate;
4. which DSV4 backend surfaces support the corresponding `M` values;
5. how requests above the captured maximum safely fall back to eager mode.

This is a preflight and attribution target. Do not promote a new default bucket
set here.

## References To Read First

Mini:

```text
performance_milestones/target12_release_max_seq_benchmark_parity/README.md
performance_milestones/target08_serving_graph_bucket_policy/README.md
performance_milestones/target08_cuda_graph_memory_attribution/README.md
performance_milestones/target08_bf16_cache_graph_memory_attribution/README.md
benchmark/offline/deepseek_v4_perf_matrix.py
python/minisgl/engine/config.py
python/minisgl/engine/engine.py
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_utils/buffers.py
/workspace/sglang-main/python/sglang/srt/server_args.py
```

vLLM:

```text
/workspace/vllm-dsv4-docker/vllm/config/vllm.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
/workspace/vllm-dsv4-docker/vllm/compilation/cuda_graph.py
```

Use source parity before inventing a mini-only policy. Record the exact source
locations and distinguish observed runtime behavior from static inference.

## Required Work

### Phase 0: Make Configuration Modes Explicit

Audit and report three distinct modes:

1. **Serving/model default:** ordinary `LLM(model_path)` semantics. Do not
   silently shrink `max_running_req` to the scenario batch size.
2. **Explicit override:** caller deliberately provides `max_running_req`, graph
   maximum, or bucket list.
3. **Scenario-sized diagnostic:** the benchmark deliberately minimizes static
   state for a focused experiment. It must be labeled diagnostic and must not
   masquerade as the release default.

For every experiment report:

```text
requested/effective max_running_req
requested/effective model max sequence length
requested/effective CUDA graph maximum and bucket list
request/context-table shape, dtype, and bytes
KV page capacity and estimated token capacity
```

The serving default currently implies many request slots; it does not imply
that all slots must simultaneously hold a 1M-token sequence.

### Phase 1: Stabilize The Real Serving Baseline

Use the model-default max sequence length, serving-default request capacity,
and current release graph buckets `[1, 2, 4, 8, 16]` as the main baseline.

- Repeat the short TP8 baseline enough to distinguish noise from a real static
  metadata or graph-memory effect.
- Explain the TARGET 12.597 one-off throughput difference between the
  model-default and scenario-sized cases before attributing it to max sequence
  length.
- Keep each engine variant in a fresh process.
- Do not silently replace the serving baseline with a scenario-sized engine if
  the real default is memory constrained. Attribute the owner instead.

### Phase 2: Source-Parity Census

Compare mini, SGLang, and vLLM for:

- bucket generation and density at small and large `M`;
- padding from an actual batch to its capture bucket;
- maximum-bucket caps and their relationship to request/token limits;
- capture order and graph-pool sharing/reuse;
- static input/output/metadata allocation;
- KV reservation before and after graph capture;
- eager behavior above the captured maximum;
- failure handling when a bucket cannot be captured.

The output must identify mechanisms worth adapting, not just list different
bucket arrays.

### Phase 3: Static And Partial-Model Shape Audit Through 512

Before loading full weights, build no-weight or one-layer probes for the DSV4
decode surfaces that scale with graph batch size:

```text
graph input/output buffers
DSV4 attention and component metadata
sampler/logit buffers
HC and dense projection paths
Marlin WNA16 MoE routing/workspaces
communication staging
request/context tables
```

At representative `M = 16, 32, 64, 128, 256, 512`, record:

```text
shape formula
dtype
allocated/live bytes
selected backend
shape guard or padding behavior
capture compatibility
```

Find allocation terms that grow with model max sequence length, request
capacity, graph maximum, or bucket count. Do not conflate allocator reserve
with live tensor ownership.

### Phase 4: Cumulative Full-Model Capture Ladder

In fresh TP8 processes, test cumulative graph maxima in this order while they
remain useful and affordable:

```text
16 -> 32 -> 64 -> 128 -> 160 -> 256 -> 512
```

Use a generated sparse policy derived from the source comparison; users should
not need to enumerate every bucket manually. For each maximum record:

```text
generated bucket list and count
capture success/failure and capture time
CUDA graph private-pool/live allocation delta
physical free-memory delta
remaining KV pages/token capacity
largest static and per-bucket owners
```

Stop increasing the practical ladder when graph memory, capture time, KV loss,
or backend behavior makes the next maximum clearly unsuitable. A `512` no-go
is an acceptable conclusion if the evidence selects a lower release maximum.

### Phase 5: Replay, Padding, And Eager Behavior

For viable buckets through 512, measure exact-bucket and padded workloads. At
minimum cover boundaries such as:

```text
17, 33, 65, 129, 257
```

Verify:

- the expected graph bucket is replayed;
- padded rows cannot affect live rows or sampled outputs;
- latency does not show an unexplained cliff at a bucket boundary;
- request waves exercise realistic admission/release behavior;
- batches above the captured maximum safely use eager execution.

Use repeated measurements for promotion-grade claims. Small/no-weight probes
may drive iteration; use full-model macro runs only for finalists.

### Phase 6: Isolated 1024 And 2048 Smoke Points

Probe `M=1024` and `M=2048` only as isolated capability points:

- static/no-weight allocation audit;
- partial-model capture if cheap and safe;
- optional full-model smoke only if prior stages leave enough memory.

Report support, dominant memory owner, capture/OOM result, and basic
correctness. Do not tune kernels, add neighboring buckets, or make these points
release blockers.

### Phase 7: Extreme 1M Plus Concurrency Smoke

Do not run an expensive many-request 1M macro. Preserve the already-proven
single-request legal 1M path, then check only that:

- configuration and admission arithmetic remain valid;
- impossible aggregate KV demand is rejected or queued cleanly;
- large short-context batches can be tested independently;
- graph policy does not falsely reserve capacity for every request slot to be
  simultaneously 1M long.

## Required Conclusion

Classify the outcome as one of:

```text
READY_FOR_TARGET_12.605
READY_FOR_TARGET_12.605_WITH_LOWER_MAX
BLOCKED_BY_GRAPH_MEMORY_OWNER
BLOCKED_BY_DSV4_BACKEND_SHAPE
BLOCKED_BY_SERVING_CAPACITY_CONTRACT
```

The report must propose an exact TARGET 12.605 contract:

```text
default generated bucket rule through the selected max <= 512
explicit override behavior
padding and eager-fallback behavior
graph-memory budget and KV-capacity guard
required correctness and repeat-stable performance gates
1024/2048 smoke-only policy
```

## Non-Goals

- Do not promote the bucket policy in this target.
- Do not require practical simultaneous 1M context and high concurrency.
- Do not tune kernels specifically for `M=1024` or `M=2048`.
- Do not change MTP, numerical precision, chunk size, page size, C128, SWA
  ownership, prefix-cache semantics, or indexer algorithms.
- Do not capture prefill graphs; this target concerns decode graph batches.
- Do not retain broad instrumentation after the evidence is recorded.

## Stop Conditions

Stop local optimization and write the report when:

- the next practical bucket is blocked by an owner outside this target;
- three focused attempts do not materially improve a confirmed local blocker;
- only `M>512` tuning remains;
- the exact 12.605 integration contract is supported by repeat-stable evidence.

## Output

```text
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
```

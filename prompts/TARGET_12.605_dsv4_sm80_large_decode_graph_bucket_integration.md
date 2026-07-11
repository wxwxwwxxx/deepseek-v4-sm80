# TARGET 12.605: DSV4 SM80 Large Decode Graph Bucket Integration

## Status

Planned after TARGET 12.604 unifies bucket resolution and reserve accounting.

## Purpose

Promote the most useful A100/sm80 DSV4 generated decode-graph policy based on
measured serving utility, graph memory, KV opportunity cost, and correctness.

`M=64` is a conservative candidate, not a predetermined answer. `M=128` must be
fairly reevaluated under the safe TARGET 12.603 reserve. Higher practical
candidates through 512 are optional only if max128 passes comfortably and a
real workload demonstrates additional value.

TARGET 12.603 established this conservative per-rank serving-default ledger at
`max_running_req=256`:

```text
max16   3,415 pages / 874,240 tokens
max64   3,314 pages / 848,384 tokens
max128  3,180 pages / 814,080 tokens
```

The conservative estimator is a valid production mechanism. Temporary
full-model profiling remains deferred until mini has an audited vLLM-like KV
detach lifecycle; do not reopen it merely to choose a bucket default.

## Inputs

```text
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
performance_milestones/target12_cuda_graph_padding_live_row_classification/README.md
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
performance_milestones/target12_cuda_graph_memory_reserve_planner/README.md
performance_milestones/target12_cuda_graph_bucket_reserve_contract_unification/README.md
```

## Required Work

### 1. Consume The Unified Policy

Use the single TARGET 12.604 resolved-policy object for planner, GraphRunner,
padding dispatch, benchmark configuration, and reports. Do not create another
generator. Preserve explicit-list override, disabled mode, and fail-closed
eager execution above the selected maximum.

Require:

```text
resolved graph tuple == estimated graph tuple == requested capture tuple
```

### 2. Carry The Existing Contracts

- Apply the TARGET 12.603 estimate plus safety margin before final KV
  allocation; report estimate, actual, remaining margin, pages/tokens, effective
  sequence width, and physical free memory.
- Preserve TARGET 12.6025: padded top-k routes are masked, one route plan is
  authoritative for Marlin, and dummy poison remains semantically inert.
- Pure exact-M versus padded-M BF16 batch-shape token non-invariance is allowed
  and documented; cross-shape token identity is not required.
- Do not silently shrink `max_running_req`, substitute scenario-sized capacity,
  or misreport a lower effective serving max as loss of model capability.

### 3. Compare Max16, Max64, And Max128

Use fresh TP8 processes and realistic short/medium-context serving waves. For
each policy record active-M distribution, resolved/padded M, graph replay/eager
counts, time and tokens in each M range, latency percentiles, aggregate
throughput, initialization time, graph memory, and KV capacity.

Explicitly test the policy transitions:

```text
max16:  M>16 uses eager
max64:  M<=64 replays; M>64 uses eager
max128: M<=128 replays
```

Include exact and padded boundaries and small-batch bs1/2/4/8/16 latency. A
synthetic always-full batch is supporting micro evidence, not sufficient proof
for a serving default.

Use at least two fresh processes per finalist after warmup and report repeat
variation. Keep benchmark lifecycle identical between policies.

### 4. Price The Capacity Tradeoff

TARGET 12.603 measured:

```text
max64 versus max16: 25,856 fewer KV tokens
max128 versus max64: 34,304 fewer KV tokens
max128 versus max16: 60,160 fewer KV tokens
```

Recompute these values from the current build and explain any difference. Judge
the larger policy by weighted serving benefit versus this opportunity cost, not
capture feasibility alone.

Keep estimator telemetry visible. Do not tighten its constants or 512 MiB
margin just to make a larger policy win. If conservative over-reservation is
the dominant remaining capacity cost, record a separate future calibration
target.

### 5. Decide Fixed Versus Memory-Aware Policy

Evaluate whether one default is sufficient or whether requested context/KV
budget should select a smaller or larger graph maximum. Ordinary
`LLM(model_path)` must obtain a safe high-performance policy without environment
variables or a manually enumerated list.

- If realistic waves rarely exceed M=64, prefer max64 or a clearly defined
  memory-aware policy.
- If M=65-128 is material and max128 gives repeat-stable benefit for its roughly
  34k-token incremental cost, max128 is a valid default candidate.
- Consider max256/512 only if max128 is comfortably positive and evidence shows
  meaningful demand. Reaching 512 is not a success criterion.

### 6. Correctness And Serving Soak

At the selected policy verify:

```text
sampler and legal token IDs
C4/C128/SWA metadata and cache ownership
Marlin MoE live-route contract
PyNCCL/NCCL communication
lm_head/output gathering
prefix hit and no-hit
request admission/release
above-max eager fallback
natural-language text sanity
```

Do not combine 1M and high concurrency as a performance gate. Preserve the
legal single-request 1M smoke separately using an appropriate low request
capacity; report the ordinary serving default's effective maximum accurately.

## Promotion Gates

- resolved, estimated, and captured bucket tuples match exactly;
- graph estimate stays within its safety contract;
- post-capture free >= 2 GiB and promotion-workload free >= 1 GiB;
- capacity loss is reported in pages/tokens;
- natural-language and cache/component lifecycle gates pass;
- repaired MoE padding contract does not regress;
- current bs1/2/4/8/16 latency does not materially regress;
- larger-M replay materially beats eager in the workload ranges it replaces;
- the selected policy has positive weighted serving value after its observed
  M distribution and KV opportunity cost are considered;
- repeat stability uses at least two fresh processes per finalist;
- above-max batches complete eagerly and observably.

## 1024/2048 Policy

Retain isolated explicit smoke tooling only. Do not add these values to the
release generator, tune their kernels, or treat failure as a release blocker.

## Required Decision

```text
PROMOTE_MEMORY_AWARE_GENERATED_BUCKETS
PROMOTE_WITH_MAX64
PROMOTE_WITH_MAX128
KEEP_MAX16_PENDING_NAMED_BLOCKER
```

Record the exact default, generation, override, fallback, graph/KV guard,
capacity, and numerical correctness contract selected.

## Output

```text
performance_milestones/target12_large_decode_graph_bucket_integration/README.md
```

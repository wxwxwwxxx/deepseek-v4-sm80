# TARGET 12.605: DSV4 SM80 CUDA Graph Recipe Frontier And Selection

## Status

Current after TARGET 12.604 unified bucket resolution and reserve accounting.

## Purpose

Measure the useful frontier between:

```text
max_running_req / max_num_seqs
cuda_graph_max_bs
graph capture memory and startup cost
KV capacity / supported context
serving replay coverage and throughput
```

Select evidence-backed recipes for DGX A100 80GB x8 style deployments. Preserve
user overrides and allow graph max below request capacity with observable eager
fallback. Do not force one graph maximum onto long-context and high-concurrency
serving.

This target selects recipes and exact promotion contracts. TARGET 12.606 owns
default wiring, cleanup, and final release soak.

## Core Contract

For current non-speculative decode, graph rows equal active decode requests:

```text
0 <= cuda_graph_max_bs <= max_running_req
```

- Equality means graph coverage can span every scheduler-legal decode batch.
- A smaller graph maximum is valid: above-max decode uses eager and trades tail
  performance for graph/KV memory efficiency.
- A graph maximum larger than request capacity adds no coverage and remains an
  invalid release configuration.
- Future speculative decoding may use a token-row contract rather than this
  request-row identity; MTP is paused and outside this target.

Do not conflate request-slot capacity with the number of simultaneous long
sequences that fit in KV memory.

## Required Inputs

```text
performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
performance_milestones/target12_cuda_graph_memory_reserve_planner/README.md
performance_milestones/target12_cuda_graph_bucket_reserve_contract_unification/README.md
prompts/archive/target12/TARGET_12.604_dsv4_sm80_cuda_graph_bucket_reserve_contract_unification.md
```

Use max-only configuration so the TARGET 12.604 resolver/generator is exercised
in every candidate. Do not manually enumerate release lists.

## Required Work

### 1. Define Three Recipe Families

Evaluate these families without creating a full Cartesian matrix.

#### Balanced Default Candidate

```text
max_running_req = 256
graph max candidates = 128, 256
short/medium contexts and mixed serving waves
```

This decides whether the ordinary DGX A100 recipe should cover all 256 request
slots or retain an eager tail above 128.

#### High-Concurrency Short/Medium-Context Candidate

```text
max_running_req = 512
graph max candidates = 256, 384, 512
explicit short/medium max sequence or KV budget
```

First run planner/no-weight or initialization feasibility. Continue to a full
serving candidate only while fixed SWA/request-table cost, graph reserve, KV
capacity, and post-capture headroom remain practical.

#### Long-Context Candidate

```text
target context = 512k
small max_running_req candidates such as 4, 8, 16, or the smallest practical set
graph max <= max_running_req, normally exact or small
```

Retain a legal 1M single-request smoke separately. Do not require high
concurrency and 512k/1M context simultaneously.

Use measured planner arithmetic to select the smallest useful request/graph
capacity rather than blindly testing every value.

### 2. Stage The Experiments

Use this order:

1. CPU/planner ledger for every proposed pair.
2. No-weight/partial initialization where it can reject an impractical pair.
3. One fresh full-model capture for surviving 384/512 candidates.
4. Repeat-stable serving runs only for finalists.

Stop a branch before full-model macro when any of these is already violated:

```text
post-capture projected free < 2 GiB
promotion-workload projected free < 1 GiB
KV capacity cannot sustain the recipe's intended context/workload
fixed request/SWA state dominates the remaining capacity
graph maximum provides no additional active-M coverage
```

Do not run max512 merely because the resolver accepts it.

### 3. Serving Workloads And Attribution

Within each fixed `max_running_req`, use identical requests, arrival/wave
pattern, prompt lengths, output budgets, seeds, and lifecycle for every graph
maximum. Record:

```text
active-M histogram and time distribution
resolved and padded M histogram
graph replay/eager counts and time by M range
request throughput and output-token throughput
TTFT / TPOT / completion latency percentiles
small-batch latency
initialization and capture time
estimate / margin / actual graph bytes
KV pages/tokens and effective max sequence
post-capture and post-workload free memory
```

At minimum, balanced waves must spend meaningful time in `M=1-16`, `17-128`,
and `129-256`. High-concurrency finalists must exercise `257-384` and/or
`385-512`; otherwise their larger graph maxima have not been evaluated.

Use exact and padded boundaries. Preserve TARGET 12.6025 dummy-route semantics;
cross-shape BF16 token identity is not required, but text sanity and state
correctness are.

### 4. Separate Graph Tradeoff From Request-Capacity Tradeoff

Only compare graph effects while `max_running_req` and workload are fixed:

```text
req256: graph128 vs graph256
req512: graph256 vs graph384 vs graph512
```

Do not attribute differences between req256 and req512 solely to CUDA graph.
The latter also changes request table, fixed SWA/component state, admission,
and available KV capacity.

Use previous max16/max64 evidence as lower-policy context; rerun them only when
needed for a common workload control.

### 5. Price Capacity And Conservative Reserve

For every candidate report exact incremental lost pages/tokens relative to the
smallest policy in the same request-capacity family. Keep TARGET 12.603's safe
estimator and 512 MiB margin unchanged during selection.

If estimator over-reservation dominates a promising recipe, record a future
calibration target. Do not tune constants until more than one candidate/process
provides stable actual data, and do not reduce safety simply to make max512 fit.

### 6. Select User-Facing Recipes

Produce exact proposed contracts for:

```text
dsv4_sm80_balanced
dsv4_sm80_high_concurrency
dsv4_sm80_long_context_512k
dsv4_sm80_1m_smoke (capability, not throughput recipe)
```

Each recipe must state:

```text
max_running_req
cuda_graph_max_bs and generated bucket rule
max sequence / KV expectation
page size and chunked-prefill assumptions
estimated/actual graph memory
expected serving regime
above-max eager behavior
known numerical contract
```

Ordinary `LLM(model_path)` should eventually receive the selected balanced
recipe automatically. Other recipes may be explicit named presets/API options;
do not require environment-variable knowledge or manual bucket lists.

### 7. Correctness And Repeat Gates

- resolved/estimated/requested/captured tuples match exactly;
- graph estimate remains within its safety contract;
- post-capture free >= 2 GiB and workload free >= 1 GiB;
- repaired MoE padding, C4/C128/SWA ownership, prefix hit/no-hit, PyNCCL,
  sampler, and output gathering remain healthy;
- natural-language Chinese/English/code/arithmetic/instruction smoke passes;
- above-max M completes eagerly and observably;
- at least two fresh processes support every promotion-grade finalist;
- weighted benefit exceeds repeat noise and is priced against lost KV capacity.

## Decision Rules

- Prefer graph128 over graph256 for the balanced recipe if M>128 is rare or
  graph256's weighted gain does not justify capacity/startup cost.
- Prefer graph256 if M=129-256 is material and coverage benefit is stable.
- Promote a req512 high-concurrency recipe only if real M>256 waves benefit and
  its context/KV capacity remains useful.
- Keep graph384/512 as explicit research configurations when they capture but
  lack practical serving ROI.
- Long-context recipes should minimize fixed request/graph state and preserve
  context capability rather than maximize decode concurrency.

## Non-Goals

- Do not promote defaults or presets in this target; TARGET 12.606 owns cleanup.
- Do not tune kernels, estimator constants, precision, MTP, page size, or chunk
  size.
- Do not require 512k/1M and high concurrency simultaneously.
- Do not optimize M>512 or add 1024/2048 to release generation.
- Do not reopen temporary full-model graph profiling.

## Required Decision

```text
RECIPE_FRONTIER_READY_FOR_12.606
KEEP_BALANCED_GRAPH128
KEEP_BALANCED_GRAPH256
HIGH_CONCURRENCY_RECIPE_NO_GO
BLOCKED_BY_CAPACITY_OR_SERVING_EVIDENCE
```

The report may combine `RECIPE_FRONTIER_READY_FOR_12.606` with the selected
balanced/high-concurrency/long-context recipe conclusions.

## Output

```text
performance_milestones/target12_cuda_graph_recipe_frontier_selection/README.md
```

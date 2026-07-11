# TARGET 12.606: DSV4 SM80 CUDA Graph Recipe Promotion And Cleanup

## Status

Current after TARGET 12.605 selected the DGX A100 recipe frontier:

```text
RECIPE_FRONTIER_READY_FOR_12.606
KEEP_BALANCED_GRAPH256
HIGH_CONCURRENCY_RECIPE_NO_GO
SELECT_LONG_CONTEXT_REQ4_GRAPH4
KEEP_1M_REQ1_GRAPH1_SMOKE
```

This target promotes recipes and establishes a provisional release baseline
for practical short/medium workloads. TARGET 12.61 owns detailed
backend/kernel attribution and any subsequent long-context kernel rewrite. A
final publication-grade performance card is measured only after that work
converges, so expensive 512K/1M numbers are not collected twice.

## Purpose

Make the selected DSV4 A100/sm80 recipes available without environment-variable
knowledge or manual bucket lists. Keep `max_running_req` and
`cuda_graph_max_bs` independently configurable, preserve explicit user
overrides and observable above-graph-max eager fallback, and publish enough
DGX A100 80GB x8 evidence for users to understand the throughput/context/memory
tradeoff.

The selected recipe families are:

| Recipe | Request capacity | Graph max | Intended use |
| --- | ---: | ---: | --- |
| `dsv4_sm80_low_m64` | 256 | 64 | latency/KV-sensitive workloads concentrated at M<=64 |
| `dsv4_sm80_mid_m128` | 256 | 128 | serving concentrated at M<=128 with more KV headroom |
| `dsv4_sm80_balanced` | 256 | 256 | throughput serving with material M=129-256; no-env default |
| `dsv4_sm80_long_context_512k` | 4 | 4 | low-concurrency 512 Ki-token serving |
| `dsv4_sm80_1m_smoke` | 1 | 1 | 1 Mi-token capability/sanity, not a throughput promise |

Do not publish a req512 high-concurrency recipe. TARGET 12.605 proved that its
fixed SWA/request state dominates memory before graph384/512 becomes useful.

## Required Inputs

```text
performance_milestones/target12_cuda_graph_recipe_frontier_selection/README.md
performance_milestones/target12_cuda_graph_bucket_reserve_contract_unification/README.md
performance_milestones/target12_moe_padding_live_route_contract_fix/README.md
performance_milestones/target12_release_max_seq_benchmark_parity/README.md
prompts/TARGET_12.605_dsv4_sm80_cuda_graph_recipe_frontier_selection.md
prompts/TARGET_12.61_dsv4_sm80_long_context_ttft_owner_attribution.md
```

## Core Contracts

### 1. Recipe And Override Contract

- Ordinary `LLM(model_path)` on the supported DSV4 A100/sm80 TP8 platform
  resolves to the balanced req256/graph256 recipe.
- Named recipes use the same public configuration resolver as the no-env
  default; do not create a second env-only configuration path.
- Explicit `max_running_req`, `cuda_graph_max_bs`, max-sequence, memory/KV, and
  chunked-prefill overrides remain authoritative after validation.
- `0 <= cuda_graph_max_bs <= max_running_req`; scheduler-legal M above graph
  max runs eagerly and is reported as such.
- Generated buckets and graph reserve are derived once before KV planning and
  are identical in planner, runner, benchmark report, and telemetry.
- Page size remains 256 and release chunked-prefill budget remains 8192 unless
  an explicit supported override is supplied.

### 2. Numerical And Correctness Contract

Preserve TARGET 12.6025 live-route padding semantics and all existing
C4/C128/SWA/component ownership contracts. Cross-bucket BF16 token identity is
not required, but every promoted recipe must pass text sanity, finite-logit,
state/lifecycle, prefix hit/no-hit, graph replay, and repeat-stability gates.

### 3. Capacity Contract

Publish effective KV pages/tokens, graph estimate/margin/actual bytes,
post-capture free memory, and total-sequence semantics with every recipe.
Balanced graph256 is expected to trade capacity for M=129-256 replay coverage:

```text
graph128 capacity: 814,080 tokens
graph256 capacity: 745,472 tokens
incremental graph256 cost: 68,608 KV tokens
```

These are measured reference values, not constants to hard-code.

## Required Work

### 1. Wire And Clean The Recipes

1. Promote req256/graph256 as the supported DGX A100 balanced default.
2. Provide explicit graph64 low-M, graph128 mid-M, and 512K/1M named recipes
   through the normal Python/API configuration surface.
3. Preserve user-selected graph maxima such as 64, 128, 256, 384, or 512 when
   valid; 384/512 remain user research configurations, not release recipes.
4. Reject graph max above request capacity and impossible memory plans with an
   actionable resolved-policy/capacity diagnostic.
5. Remove stale promotion-only env gates and broad instrumentation that no
   longer protects a fallback/oracle. Keep narrow debug counters opt-in where
   they remain useful.
6. Document cold capture/startup cost. TARGET 12.605 measured about 45.6 s for
   a cold graph256 process and about 18.8 s with warm compile caches.

### 2. Build A DGX A100 Provisional Performance Card

Use fresh TP8 processes, the true no-env release path or named public recipe,
fixed seeds, page size 256, chunk size 8192, and repeat-stable timing. Report
hardware/software revisions and both scheduler M and resolved/padded M.

The requested serving grid is a specification table, not permission to exceed
physical KV capacity:

```text
active decode M:       4, 16, 64, 128, 256
prompt per request:    1K, 4K, 16K
decode per request:    1K
```

First compute the full Cartesian planner ledger. Classify each cell as:

```text
RUNNABLE
SKIP_CAPACITY_NO_GO
BLOCKED_BY_KERNEL_OR_CORRECTNESS
```

Execute all `RUNNABLE` cells needed to cover every M and prompt-length axis,
but do not force impossible cells by silently changing max sequence, decode
length, request count, KV budget, or graph reserve. For skipped cells, publish
the exact required tokens, available tokens, and reason.

A performance cell is valid only when all requested sequences fit
simultaneously and the requested M rows are resident and actively decoding.
Do not use scheduler pending time, staggered admission, or multiple waves to
manufacture TTFT/TPOT/throughput for a cell that cannot coexist. Such a cell is
simply skipped; retain only its planner/capacity ledger.

For each executed cell record at least:

```text
request and output-token throughput
prefill and decode token throughput
TTFT / TPOT / inter-token or completion latency percentiles
active/resolved/padded M histogram
graph replay/eager counts and time
initialization/capture time
KV pages/tokens and peak/post-workload memory
text sanity and finite-logit result
```

Run at least two fresh processes for the balanced default's representative
cells. A single warm process is not sufficient promotion evidence.

### 3. Long-Context Rows

Keep request-slot capacity distinct from simultaneous resident long requests.
`req4/graph4` means the engine has four slots; it does not imply four independent
512K sequences fit in the measured 1.636M-token KV capacity.

For recipe promotion, run only bounded correctness/capability smoke and
classify the larger combinations:

1. **512 Ki capability, single active long request:** use a legal prompt plus a
   short bounded decode (for example `524,280 + 8 = 524,288`) under req4/graph4.
   Reuse TARGET 12.605 evidence if no relevant implementation changed.
2. **Aggregate 512 Ki, bs=4:** planner-classify four requests with
   `prompt=130,048`, `decode=1,024` each, for exactly 524,288 aggregate resident
   tokens. Defer its performance run until after TARGET 12.61 unless it is
   needed to diagnose a promotion blocker.
3. **Four 512K requests:** planner/capacity classification only unless a future
   recipe proves at least about 2.0M resident tokens. It is expected NO-GO here.
4. **1 Mi capability, bs=1:** run one legal prompt plus short bounded decode
   under req1/graph1, for example `1,048,568 + 8 = 1,048,576`. This is a
   capability/sanity run, not a production throughput measurement.

The current graph1 report only priced a short request and reused an older 1M
proof with a different graph policy. Therefore run one actual legal 1M
req1/graph1 short-decode smoke before promotion, unless an explicit time/cost
stop is recorded as the only remaining release evidence gap. Do not run a 1K
decode here merely to populate a table that may change after kernel tuning.

### 4. Final Promotion Soak

Include:

- historical bs4 / 4096x128 control and the TARGET 12.605 mixed balanced wave;
- low-M-heavy serving under graph64/128 and graph256;
- a material M=129-256 wave under graph256;
- prefix hit/no-hit, SWA independent lifecycle, C4/C128, PyNCCL, sampler, and
  output gathering;
- Chinese/English/code/arithmetic/instruction text smoke;
- above-max eager behavior for a lower-graph-max explicit configuration;
- capacity and graph reserve telemetry equality;
- bounded 512K/1M capability gates above, without publication-grade timing.

Do not rerun every historical development oracle. Use focused tests for code
that changed and retain one release-level integrated soak.

### 5. Hand Off Kernel Evidence To TARGET 12.61

The provisional performance card must identify anomalous or dominant owners,
but this target must not rewrite kernels. Leave a table with measured latency
share, scaling slope, backend, and expected upside for TARGET 12.61.

Capacity-impossible cells are omitted from the observed performance table and
listed only in a separate skipped-capacity ledger. Correctness-blocked cells
remain blockers, not performance measurements. Never report pending/admission
time or an estimate as an observed result.

## Promotion Rules

Promote the balanced default only when:

- representative gain exceeds fresh-process noise;
- historical small-M/bs4 behavior is neutral within noise or explained;
- graph estimate/reserve/actual and effective KV capacity are visible;
- all integrated correctness and lifecycle gates pass;
- no hidden env setup or manual bucket list is required.

Named 512K/1M recipes require sanity and capacity correctness. They do not need
to match balanced throughput, and extreme long-context plus high concurrency
is not a release requirement. Their final decode-1K performance rows are
published only after TARGET 12.61 and any evidence-backed kernel work converge.

## Stop Conditions

- Stop tuning any individual kernel in this target; transfer evidence to
  TARGET 12.61.
- Stop a Cartesian cell before loading weights when planner arithmetic proves
  it cannot fit.
- Do not make req512, graph384/512, or four concurrent 512K requests pass by
  reducing safety margin or misreporting available KV.
- Do not spend more than one focused rerun on a non-material performance delta
  below repeat noise.

## Required Decision

```text
PROMOTE_BALANCED_REQ256_GRAPH256
PROMOTE_NAMED_GRAPH64_GRAPH128_AND_LONG_CONTEXT_RECIPES
KEEP_1M_AS_CAPABILITY_SMOKE
BLOCKED_BY_DEFAULT_CORRECTNESS_OR_CAPACITY
HAND_OFF_LONG_CONTEXT_OR_LARGE_M_KERNEL_OWNER_TO_12.61
```

## Output

```text
performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md
```

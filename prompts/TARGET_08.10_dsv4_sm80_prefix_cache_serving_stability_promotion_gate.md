# TARGET 08.10: DSV4 Prefix Cache Serving Stability And Promotion Gate

## Status

Planned after TARGET 08.05 and TARGET 08.06.

This target uses the serving workload and CUDA graph bucket policy selected by
TARGET 08.05, plus the graph memory conclusion from TARGET 08.06, to decide
whether the DSV4 radix prefix cache can move from phase-1 experimental opt-in
to a controlled/promotable path.

## Goal

Stress and validate the current DSV4 radix prefix cache under serving-like
conditions.

The target should answer:

1. Is the phase-1 full-page-owner prefix cache stable under sustained
   multi-request workloads?
2. Does it preserve correctness under hit/miss/partial-hit/eviction pressure?
3. Does it produce repeat-stable TTFT and prefill-forward wins when graph bucket
   coverage is configured correctly?
4. Is it safe to promote as a controlled opt-in or default for DSV4?

## Required Starting State

Read:

- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.05_dsv4_sm80_serving_workload_cuda_graph_bucket_policy.md`
- `prompts/TARGET_08.06_dsv4_sm80_cuda_graph_memory_attribution.md`
- `performance_milestones/target08_radix_prefix_dsv4/README.md`
- `performance_milestones/target08_radix_prefix_dsv4/DESIGN.md`
- TARGET 08.05 result README after it exists.
- TARGET 08.06 result README after it exists.

Use:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
--page-size 256
--num-pages 128 or a justified capped value
--enable-dsv4-radix-prefix-cache for the opt-in run
```

Keep `page_size % 128 == 0` as a required DSV4 radix-prefix constraint unless
the target finds a stronger vLLM/SGLang-aligned safe alignment rule.

## Workloads

Use the TARGET 08.05 recommended graph bucket policy.
Carry forward the TARGET 08.06 graph memory conclusion when deciding whether
the feature is promotable or should remain controlled opt-in.

At minimum test:

- prefix disabled control;
- prefix enabled;
- one repeated shared-prefix workload;
- one mixed hit/miss workload;
- one partial-hit workload;
- one sustained multi-prefix workload;
- one eviction-pressure workload using fewer pages or enough distinct prefixes
  to force evictions safely.

Prefer `requests >= 100` for the sustained serving-style run if runtime allows.
If not, document why and use the largest practical substitute.

## Correctness Gates

Run targeted tests or smoke scripts for:

- full hit;
- partial hit;
- miss;
- repeated hit/evict cycle;
- eviction under pressure;
- multiple prefix keys;
- SWA boundary at and around `128`;
- C4/C128/indexer component boundaries;
- generated text or logits against prefix-disabled mode.

Graph replay must remain active for the bucket sizes selected by TARGET 08.05.
Eager decode should be explained by bucket policy, not by prefix-cache breakage.

## Metrics

Report:

- prefix hit rate;
- full/partial/miss request counts;
- saved prefill tokens;
- suffix prefill tokens after hit;
- retained prefix pages/tokens;
- retained DSV4 full/C4/C128/C4-indexer/state slots;
- estimated retained DSV4 memory;
- evictions and evicted pages/tokens;
- TTFT, ITL/TPOT if available;
- prefill-forward and decode-forward time;
- output tok/s and decode tok/s;
- graph replay/eager counts by batch size;
- peak allocated/reserved memory;
- failures, retries, or correctness mismatches.

## Deliverables

Create:

```text
performance_milestones/target08_prefix_cache_serving_stability/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- design recap from phase 1;
- exact commands;
- graph bucket policy used;
- correctness table;
- serving workload table;
- eviction pressure result;
- memory retention table;
- promote/controlled-opt-in/reject decision;
- whether TARGET 08.18 should be run before any default promotion.

## Promotion Rules

Promote at most to a controlled opt-in if all are true:

- correctness gates pass;
- no DSV4 component leak or double-free;
- graph replay coverage matches TARGET 08.05 policy;
- TTFT/prefill-forward improvements are repeat-stable on shared-prefix
  workloads;
- memory retention is understood;
- eviction pressure does not cause unexplained correctness or latency failures.

Do not promote to unconditional default in this target if memory retention has
not been analyzed under long-prefix and multi-prefix capacity pressure.  That
analysis belongs to TARGET 08.18.

## Stop Rules

Stop and report blocked if:

- prefix-enabled output diverges from prefix-disabled output;
- SWA/C4/C128/indexer refcount integrity fails;
- eviction causes leaks or double-free;
- graph replay is unexpectedly disabled;
- memory retention causes OOM under the required fixed/capped page policy;
- benchmark noise prevents a stability decision.

## Non-Goals

- SGLang-style independent SWA component implementation.
- Low-precision KV cache or INT8 MoE.
- Attention-kernel optimization.
- PyNCCL or communication overlap tuning.

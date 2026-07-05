# TARGET 09.4: DSV4 SM80 Minimal FP8 KV/Cache Slice

## Status

Conditional.  Run only if TARGET 09.3 identifies a concrete FP8 cache slice with
meaningful capacity or bandwidth upside.

## Goal

Implement and benchmark the smallest FP8 KV/cache slice that exercises the real
store/quant and gather/dequant boundary.

This target should prove whether the selected FP8 cache design is practical
before attempting full E2E rollout.

## Required Design Constraints

- Follow SGLang/vLLM layout and scale behavior unless there is hard evidence for
  a better mini-specific design.
- Avoid standalone quant/dequant paths that cause extra HBM round trips.
- Keep prefix-cache Route B component ownership correct.
- Keep graph replay compatible.
- Keep feature opt-in and rollback simple.

## Required Work

1. Implement minimal slice

   Implement only the selected component from TARGET 09.3.  Examples might be:

   - compressed/MLA KV store plus gather/dequant;
   - indexer/cache component store plus gather/dequant;
   - a decode-only slice if prefill path is too broad.

   Do not broaden the slice without updating the report.

2. Microbench

   Compare:

   - current BF16 cache path;
   - FP8 store/quant;
   - FP8 gather/dequant;
   - combined store + decode gather/dequant at real shapes.

   Include HBM traffic estimates and scale/workspace overhead.

3. Correctness checks

   Run selected-row/page comparisons against BF16 cache:

   - direct cache value checks after store;
   - gather/dequant value checks;
   - attention input checks if practical;
   - prefix-cache hit/remap checks for touched components.

4. Graph checks

   Verify graph capture and replay for relevant batch buckets.  Record eager
   fallbacks and graph memory delta.

5. Decision

   Decide whether TARGET 09.5 is justified.

## Gates

Pass if:

- FP8 slice is source-aligned with SGLang/vLLM or explicitly justified;
- microbench shows useful capacity and no unacceptable latency loss;
- correctness checks pass within documented tolerance;
- graph replay remains active;
- capacity ledger remains meaningful after scales/workspace.

Stop if:

- store/quant is slow on SM80;
- gather/dequant requires full-cache dequantization;
- standalone casts dominate decode;
- prefix-cache correctness breaks;
- graph replay breaks;
- macro ceiling is below `1%` and capacity savings are not compelling.

## Deliverables

Write results under:

```text
performance_milestones/target09_minimal_fp8_kv_cache_slice/
```

Include:

- `README.md` with TARGET 09.5 go/no-go;
- implementation notes and env flags;
- microbench tables;
- correctness outputs;
- graph replay/capture status;
- capacity ledger update;
- rollback instructions.


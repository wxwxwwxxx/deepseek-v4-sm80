# TARGET 09.5: DSV4 SM80 FP8 KV/Cache Opt-In E2E

## Status

Conditional.  Run only if TARGET 09.45 selects an FP8 KV/cache E2E scope after
recomputing memory ROI, speed ROI, and SGLang-aligned SWA lifecycle effects.

TARGET 09.4 proved the SWA packed MLA boundary is correct and graph-safe, but
its separated store/gather kernels were slower than the BF16 boundary.  Do not
assume SWA-only E2E is the next step unless TARGET 09.45 confirms that its
capacity value survives the SWA lifecycle reset.

## Goal

Integrate the FP8 KV/cache path selected by TARGET 09.45 as an opt-in E2E
feature and classify it as a throughput feature, capacity feature, or rejected
experiment.

## Opt-In Surface

Prefer a clear cache-format flag:

```text
MINISGL_DSV4_SM80_FP8_MLA_KV_CACHE=1
MINISGL_DSV4_SM80_FP8_KV_CACHE_LAYOUT=sglang_scaled
```

If a general `--kv-cache-dtype` interface exists by then, map this feature into
that public surface and avoid duplicate knobs.

## Required Work

1. Full selected-path integration

   Extend only the component set selected by TARGET 09.45.  Do not expand to
   unrelated cache components unless the report updates the capacity, speed, and
   correctness gates.

2. Correctness and quality

   Run:

   - generated-text smoke;
   - fixed-prompt logit diff or top-k stability;
   - prefix-cache verifier;
   - selected cache row/page value checks;
   - serving-style smoke with prefix cache enabled.

3. Performance

   Compare against the TARGET 10 baseline:

   - `historical_4096_128_bs4`;
   - `historical_4096_1024_bs4`;
   - `serving_mixed_112req_wave16`;
   - `prefix_multi_112req_wave16`;
   - any longer-context or larger-page scenario justified by the capacity win.

   Capture owner timing, graph replay, and peak memory.

4. Capacity ledger

   Report:

   - bytes/token before and after;
   - GiB/rank saved;
   - additional pages/tokens possible;
   - graph capture headroom;
   - serving capacity implication.

5. Promotion decision

   Decide whether the feature should remain opt-in, become a capacity mode, or
   be rejected.

## Gates

Pass if:

- text smoke and prefix-cache verifier pass;
- quality drift is measured and acceptable;
- graph replay remains active;
- capacity win is meaningful after scales/workspace;
- throughput regression is acceptable for a capacity mode, or throughput
  improves by at least `3%` for a throughput mode;
- rollback is one env/variant change.

Stop if:

- quality drift is unexplained;
- prefix-cache correctness breaks;
- graph replay breaks;
- memory saved is too small;
- throughput regression is material and there is no capacity justification.

## Deliverables

Write results under:

```text
performance_milestones/target09_fp8_kv_cache_optin_e2e/
```

Include:

- `README.md` with promotion recommendation;
- exact commands/env;
- correctness reports;
- macro and owner timing reports;
- capacity ledger;
- known limitations;
- rollback instructions.

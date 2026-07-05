# TARGET 09.6: DSV4 SM80 Quantized Projection / Cache-Boundary Fusion

## Status

Optional later target.

Run this only if a fresh profile shows projection or cache-boundary HBM traffic
is again material after TARGET 07/08/10 and any earlier TARGET 09 work.

## Goal

Decide whether moving quantization into projection or cache-store kernels can
reduce HBM traffic enough to beat the current promoted path.

This is lower priority than INT8 MoE and FP8 KV/cache unless profiles say
otherwise.

## Candidate Areas

- Dense projection GEMM with internal activation quantization.
- Projection weight/cache boundary that currently writes BF16 intermediates.
- Cache-store kernels that can pack/quantize without standalone casts.
- Gather/dequant kernels that can fuse dequant with consumer computation.

## Required Work

1. Fresh profile proof

   Show the target owner is material.  Do not optimize this area from intuition
   alone.

2. Source parity

   Check SGLang/vLLM for existing kernels or layout decisions before writing a
   mini-owned implementation.

3. HBM traffic model

   For each candidate, estimate:

   - bytes read and written today;
   - bytes read and written after fusion;
   - scale/workspace overhead;
   - expected A100 roofline limit;
   - maximum possible owner-time win.

4. Microbench

   Implement or port only the smallest candidate needed to test the traffic
   model.  Compare against current cached BF16/Marlin paths.

5. E2E only if justified

   Run full E2E only if microbench and owner replay predict a meaningful macro
   win.

## Gates

Pass if:

- source parity is checked;
- HBM model predicts a real win;
- microbench confirms the model;
- graph replay remains compatible;
- correctness smoke passes if a runtime path is integrated.

Stop if:

- owner time is no longer material;
- candidate duplicates a slower version of an existing SGLang/vLLM path;
- standalone quant/dequant overhead dominates;
- expected E2E gain is below `1%`;
- implementation starts becoming a broad rewrite.

## Deliverables

Write results under:

```text
performance_milestones/target09_quantized_projection_cache_boundary_fusion/
```

Include:

- `README.md` with continue/stop recommendation;
- owner timing proof;
- source-parity notes;
- HBM traffic and roofline model;
- microbench results;
- any E2E results if run;
- rollback instructions.


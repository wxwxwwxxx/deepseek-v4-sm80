# TARGET 09.2: DSV4 SM80 INT8 MoE Opt-In Integration

## Status

Conditional.  Run only if TARGET 09.1 proves an INT8 W8A8 backend is real,
correct enough, and likely faster than the current Marlin WNA16 MoE path.

## Goal

Integrate an opt-in INT8 MoE path for DSV4 routed experts and evaluate it
against the promoted TARGET 10 baseline.

This target may produce a rejected opt-in.  Do not promote by default unless the
gates are clearly met.

## Required Opt-In Boundary

Use one explicit opt-in surface.  Prefer:

```text
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=int8_w8a8_marlin
MINISGL_DSV4_SM80_INT8_MOE=1
```

Do not create multiple aliases unless one already exists and must be kept for
compatibility.

## Design Constraints

- INT8/INT32 values stay inside the expert backend.
- MoE TP reduce input remains BF16 or FP32.
- Do not change TARGET 10 PyNCCL defaults in this target.
- Preserve a one-command rollback to the promoted exact path.
- If original FP4 GPU expert weights are not needed after INT8 packing, release
  them or document why they must stay resident.
- Keep graph-captured decode compatible with the target batch buckets.

## Required Work

1. Integrate backend

   Wire the selected backend into the DSV4 routed expert path.  Keep the current
   Marlin WNA16 path available.

2. Weight preparation

   Convert FP4/MXFP4 expert weights to the selected INT8 backend layout during
   model load or explicit preparation.  Record:

   - load-time conversion cost;
   - peak and steady-state memory;
   - whether original FP4 expert weights are freed;
   - rollback behavior.

3. Activation quantization

   Implement the best boundary from TARGET 09.1.  Prefer fused quantization.
   If any standalone quant/dequant remains, attribute its HBM and kernel cost.

4. Correctness gates

   Run:

   - kernel/oracle checks against current exact path;
   - generated-text smoke;
   - logit diff or top-k stability on fixed prompts;
   - prefix-cache smoke when prefix cache is enabled.

5. Performance gates

   Compare against:

   ```text
   dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
   ```

   Run at least:

   - `historical_4096_128_bs4`;
   - `historical_4096_1024_bs4`;
   - `serving_mixed_112req_wave16`;
   - `prefix_multi_112req_wave16` if prefix/cache paths are touched.

   Capture owner timing for MoE, communication, metadata, graph replay, and
   memory.

## Gates

Pass if:

- text smoke passes;
- quality drift is measured and acceptable;
- graph replay remains active for target buckets;
- MoE owner time improves by at least `5%`;
- E2E improves by at least `3%` on dominant scenarios, or a documented capacity
  win is worth an opt-in;
- rollback is one env/variant change.

Stop if:

- backend falls back silently;
- TP reduce boundary becomes INT8 without TARGET 09.25 evidence;
- graph replay is broken;
- quant/dequant overhead erases MoE savings;
- E2E is neutral or negative after repeat-stable measurement.

## Deliverables

Write results under:

```text
performance_milestones/target09_int8_moe_optin_integration/
```

Include:

- `README.md` with promote/keep-opt-in/reject recommendation;
- exact env/variant commands;
- correctness outputs;
- owner timing before/after;
- macro results;
- memory ledger;
- rollback instructions.


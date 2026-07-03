# TARGET 09: DSV4 SM80 Low-Precision Research Roadmap

## Status

Planned after TARGET 08.

Do not run this before DSV4 radix/SWA prefix cache has a correctness and
performance baseline.  TARGET 09 is intentionally separated from the exact
TARGET 07 path because it may change activation, cache, or expert precision and
therefore needs stronger quality gates.

## Goal

Evaluate whether lower-precision runtime paths can beat the promoted exact-ish
SM80 path without unacceptable quality loss.

The guiding rule is:

```text
Align with vLLM's proven behavior first unless there is clear evidence that a
mini-owned alternative is simpler, correct, and faster.
```

## Starting Knowledge

Already known from TARGET 07:

- FP8 indexer/cache pieces can be useful when the backend matches vLLM's actual
  implementation.
- A naive mini-owned FP8 indexer/logits path was slower.
- Full `fp8_ds_mla` KV cache was deferred because SM80 store/quant, layout,
  gather/dequant, graph capture, and quality gates are broader than one kernel.
- Dense FP8 Marlin projection is speed-neutral on the promoted path but saves
  about `807 MB/rank`.
- INT8 W8A8 projection experiments were not selected for dense projection.
- INT8 MoE remains a research option, but it is a precision-risk path and must
  be opt-in.

## Candidate Lanes

### Lane A: vLLM-Aligned FP8 KV / Indexer / MLA Cache

Questions:

- Which exact vLLM `deepseek_v4_fp8` cache pieces run on SM80?
- Where does vLLM quantize: inside store kernels, inside attention kernels, or
  around cache boundaries?
- Can mini adopt the same layout without breaking DSV4 prefix cache?
- Does FP8 KV reduce memory enough to increase useful context capacity?
- Does it improve TTFT/decode after TARGET 08 changes the prefill profile?

Gate:

- standalone backend parity with vLLM;
- TP8 smoke;
- prefix-cache compatibility check;
- graph replay remains active;
- measurable E2E or capacity win.

### Lane B: INT8 MoE W8A8 Opt-In

Questions:

- Can routed experts use INT8 Tensor Core paths on A100 more efficiently than
  current Marlin WNA16 MXFP4 expert backend?
- What activation quantization is required?
- Can quantization be fused into dispatch/GEMM boundaries instead of adding
  slow standalone quant/dequant kernels?
- What logit/token quality delta appears on smoke/oracle prompts?

Gate:

- opt-in only;
- no silent fallback;
- compare against current Marlin WNA16 exact MoE baseline;
- correctness oracle plus generated-text smoke;
- logit difference or top-k stability report;
- macro gain large enough to justify precision risk.

### Lane C: Quantized Projection / Cache Boundary Fusion

Questions:

- Can quantization be moved inside GEMM or cache-store kernels to avoid extra
  HBM reads/writes?
- Are there vLLM kernels worth porting into mini's ABI?
- Does the route beat cached BF16 projection weights on real TP8 decode, not
  just standalone M=4 GEMMs?

Gate:

- prove standalone backend speed first;
- include HBM traffic and workspace accounting;
- require repeat-stable macro gain;
- keep memory-only wins as memory opt-ins.

## Required Quality Gates

Every low-precision candidate must report:

- exact precision boundary;
- whether weights, activations, KV cache, indexer cache, or logits changed;
- source of scales and quantization algorithm;
- vLLM parity status;
- text smoke result;
- logit diff or top-k stability where feasible;
- graph replay/eager decode status;
- memory capacity delta;
- TP8 macro result with repeat-stable comparison.

## Promotion Rules

Promote a low-precision feature only if all are true:

- correctness/quality gates pass;
- no silent fallback;
- graph replay remains active;
- repeat-stable E2E gain is at least `3%`, or the feature is explicitly marked
  as a memory/capacity mode;
- memory/workspace accounting is documented;
- a rollback env/variant exists.

## Stop Rules

Stop a lane if:

- the required backend does not support SM80 and cannot be built into mini's ABI
  within the target;
- standalone quant/dequant overhead dominates the expected win;
- quality drift is unexplained;
- prefix cache compatibility is broken;
- the macro result is neutral or negative after a fair repeat-stable gate.

## Non-Goals

- Replacing TARGET 08 prefix cache.
- Changing the promoted exact bundle by default.
- Running broad quality evaluation beyond the smoke/oracle gates needed for an
  engineering decision.
- Porting large vLLM subsystems wholesale without isolating the actual backend
  boundary that provides the win.

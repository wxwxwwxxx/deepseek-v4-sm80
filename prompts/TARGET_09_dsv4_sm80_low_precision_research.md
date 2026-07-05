# TARGET 09: DSV4 SM80 Low-Precision Research Roadmap

## Status

Recommended next family after TARGET 10.

TARGET 07 established the exact-ish A100/sm80 victory path, TARGET 08 added the
prefix-cache baseline, and TARGET 10 default-promoted the communication path:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL enabled by default for that preset
Default DSV4 sm80 PyNCCL max buffer size: 32M unless
MINISGL_PYNCCL_MAX_BUFFER_SIZE is explicitly set.
```

TARGET 09 is intentionally separate from the exact TARGET 07/08/10 route
because it may change activation, cache, or expert precision and therefore
needs stronger quality gates.

Current decision from TARGET 09.45: do not run TARGET 09.5 yet.  First run
TARGET 08.31 to prove SGLang-aligned independent SWA lifecycle and real SWA
tail-page occupancy.  That lifecycle changes the memory denominator for
SWA-only FP8 cache.

TARGET 08.32 can run in parallel as a capacity/headroom investigation for CUDA
graph private-pool memory.  It is not a precision prerequisite for TARGET 09.5,
but its outcome may change how much value remains in FP8 cache capacity work.

## Goal

Evaluate whether lower-precision runtime paths can beat the promoted
A100/sm80 DSV4 path without unacceptable quality loss.

Two primary lanes matter now:

1. INT8 MoE W8A8: convert model-native FP4 expert weights to an INT8
   backend-specific layout on load, quantize activations at the right MoE
   boundary, and try to run routed experts on A100 INT8 Tensor Cores.
2. FP8 KV/cache: align with SGLang/vLLM DeepSeek V4 cache layouts and determine
   whether storing selected KV/cache components as FP8 is worth the cast,
   gather/dequant, graph-capture, correctness, and prefix-cache complexity.

The guiding rule is:

```text
Use SGLang/vLLM as implementation oracles first.  If mini uses a different
algorithm, prove why it is simpler, correct, and faster before integrating it.
```

## Current Baseline

Use the post-TARGET-10 promoted preset as the default comparison point:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL default threshold32m
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Minimum macro scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16` when cache/prefix correctness or capacity is
  affected.

## Starting Knowledge

Already known from TARGET 07:

- The winning exact MoE route was not INT8 activation quantization; it was a
  model-native MXFP4/WNA16 Marlin expert backend.
- Dense FP8 Marlin projection is speed-neutral on the promoted path but saves
  about `807 MB/rank`; treat it as a memory/capacity feature unless a later
  profile changes the bottleneck.
- FP8 indexer/cache pieces can be useful only when the backend matches vLLM or
  SGLang's actual implementation.
- A naive mini-owned FP8 indexer/logits path was slower.
- Full `fp8_ds_mla` KV cache was deferred because SM80 store/quant, layout,
  gather/dequant, graph capture, and quality gates are broader than one kernel.
- INT8 W8A8 dense projection experiments were not selected for dense projection.
- INT8 MoE remains a research option, but it is a precision-risk path and must
  be opt-in.

Already known from TARGET 10:

- Communication path changes are now defaulted: PyNCCL threshold32m is part of
  the promoted A100/sm80 path.
- Do not mix low-precision experiments with new communication routing unless a
  fresh profile proves communication is again the bottleneck.

## Reference Implementations

Primary mini surfaces:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/marlin_wna16.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/kernel/triton/fused_moe.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`

SGLang references:

- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/`
- `/workspace/sglang-main/python/sglang/jit_kernel/moe_wna16_marlin.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/per_token_group_quant_8bit.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/gemm/marlin_moe/`

vLLM references:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops.py`

vLLM runtime remains available through:

```text
/workspace/venvs/vllm-dsv4/bin/python
```

Mini uses the system Python from `/workspace/mini-sglang`.

## Split Plan

Run in this order unless a fresh profile strongly changes the bottleneck:

| Stage | Prompt | Status | Purpose |
| --- | --- | --- | --- |
| TARGET 09.0 | `prompts/TARGET_09.0_dsv4_sm80_low_precision_preflight.md` | ready | Post-TARGET-10 low-precision preflight: fresh profile, module ownership, memory ledger, and source census for INT8 MoE and FP8 KV/cache. |
| TARGET 09.1 | `prompts/TARGET_09.1_dsv4_sm80_int8_moe_backend_feasibility.md` | ready | INT8 MoE backend feasibility: identify a real SM80 backend, weight conversion layout, activation quantization boundary, and microbench/no-weight gates. |
| TARGET 09.2 | `prompts/TARGET_09.2_dsv4_sm80_int8_moe_optin_integration.md` | conditional | INT8 MoE opt-in integration if 09.1 proves backend and quantization overhead can beat exact Marlin WNA16. |
| TARGET 09.25 | `prompts/TARGET_09.25_dsv4_sm80_int8_comm_boundary_feasibility.md` | optional research | INT8 communication feasibility for TP reduce boundaries; prove scale/overflow semantics and microbench value before any E2E integration. |
| TARGET 09.3 | `prompts/TARGET_09.3_dsv4_sm80_fp8_kv_cache_parity_ledger.md` | ready | FP8 KV/cache source parity and capacity ledger: map SGLang/vLLM layouts, cast points, components, and prefix/graph constraints. |
| TARGET 09.4 | `prompts/TARGET_09.4_dsv4_sm80_minimal_fp8_kv_cache_slice.md` | conditional | Minimal FP8 KV/cache slice: store/quant + gather/dequant microbench against mini BF16 cache and SGLang/vLLM behavior. |
| TARGET 09.45 | `prompts/TARGET_09.45_dsv4_sm80_fp8_cache_roi_sglang_lifecycle.md` | completed decision | FP8 cache ROI reset selected lifecycle-first: run TARGET 08.31 before any FP8 cache E2E. |
| TARGET 08.31 | `prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md` | next before 09.5 | SGLang-aligned SWA independent lifecycle; prove real SWA tail pages and memory headroom before reopening FP8 cache. |
| TARGET 09.5 | `prompts/TARGET_09.5_dsv4_sm80_fp8_kv_cache_optin_e2e.md` | deferred pending 08.31 | FP8 KV/cache opt-in E2E only if 08.31 shows enough real SWA/cache value or motivates a broader MLA/indexer FP8 scope. |
| TARGET 09.6 | `prompts/TARGET_09.6_dsv4_sm80_quantized_projection_cache_boundary_fusion.md` | optional later | Quantized projection/cache-boundary fusion only if fresh evidence shows projection/cache traffic is again material. |

Do not implement INT8 MoE and FP8 KV/cache in the same child thread.  They have
different correctness, memory, and graph-capture failure modes.

Do not combine INT8 MoE compute integration with INT8 communication unless
TARGET 09.25 has already proved the communication boundary is mathematically
safe and faster than the promoted BF16 PyNCCL reduce path.

## Lane A: INT8 MoE W8A8 Opt-In

### Hypothesis

The model's routed expert weights are model-native FP4/MXFP4.  On A100/sm80,
there are no native FP4 Tensor Cores, so the exact path currently uses a
WNA16/Marlin-style route that dequantizes weights internally and computes with
BF16 activations.  An opt-in INT8 route may be faster if:

- expert weights can be converted on load from FP4/MXFP4 to an INT8
  backend-specific packed layout;
- routed activations can be quantized to INT8 at a fused dispatch or
  activation boundary;
- W13/up-gate and W2/down GEMMs use a real grouped MoE INT8 Tensor Core backend;
- output dequant/finalization does not erase the Tensor Core win.

### Current Mini TP / Reduce Boundary

Mini's current DSV4 routed-expert path is tensor-parallel over the expert
intermediate dimension, not expert-parallel over expert ids:

- `DSV4FusedRoutedExperts` computes
  `local_intermediate = div_even(config.moe_intermediate_size, tp.size)`;
- each TP rank owns all routed experts, but only the local intermediate shard of
  W13/W2;
- routed experts therefore produce a partial `[tokens, hidden]` output on each
  rank;
- the promoted runner computes routed experts and shared experts with
  `reduce=False`, adds them locally, then does one final
  `dsv4.v1_moe_reduce_once_all_reduce`;
- the TARGET 10 promoted preset casts that combined MoE output to BF16 before
  PyNCCL all-reduce via `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`;
- mini's PyNCCL wrapper currently maps only FP16, BF16, and FP32 tensors to NCCL.

Therefore the first INT8 MoE design should keep INT8/INT32 values inside the
expert backend and dequantize/finalize to BF16 or FP32 before the TP reduce.  Do
not send INT8 tensors through PyNCCL/NCCL unless a separate communication target
proves both dtype support and scale/overflow correctness.

### Required Source Questions

Before writing a large integration:

- Which SGLang or vLLM backend can actually run grouped MoE W8A8 on SM80?
- Does that backend support the DSV4 expert shapes, top-k, group routing, and
  local expert layout?
- Does it expect per-tensor, per-channel, per-group, or per-token scales?
- Can model-native FP4/MXFP4 weights be converted directly to the backend's
  INT8 layout at load time, without a repeated FP4 -> BF16 -> INT8 runtime
  round trip?
- Where should activation quantization happen:
  - after route/dispatch before W13;
  - fused with route dispatch;
  - fused with SiLU/mul before W2;
  - inside the INT8 GEMM backend itself?
- What dtype does the candidate backend emit at the MoE TP boundary:
  BF16/FP32 partial hidden output, INT32 accumulator, or INT8 scaled output?
- If it proposes INT8/INT32 communication, what are the exact scale semantics,
  accumulation range, overflow behavior, and PyNCCL/NCCL dtype support?
- Can original FP4 GPU expert weights be released after INT8 packed weights are
  prepared, while still keeping a rollback path for correctness comparison?

### Backend Rules

Do not use a slow generic implementation as the candidate just because it is
easy to wire.

Acceptable first candidates:

- a SGLang/vLLM Marlin or grouped MoE backend that explicitly supports INT8
  W8A8 on SM80;
- a mini-owned bridge compiled against mini's Torch ABI if the source backend is
  real but package/ABI integration is the blocker;
- a standalone backend microbench that proves the Tensor Core path before E2E.

Unacceptable as a performance candidate:

- `torch` fallback INT8 matmul without grouped MoE support;
- separate `dispatch -> standalone quant -> GEMM -> standalone dequant` if the
  standalone casts dominate the expected win;
- a path that silently falls back to BF16/FP32 while reporting INT8.

Standalone quant/dequant is allowed only as an oracle or attribution tool.

### Suggested Opt-In Surface

Prefer explicit names that cannot accidentally become default:

```text
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=int8_w8a8_marlin
MINISGL_DSV4_SM80_INT8_MOE=1
```

Use one canonical env/variant after implementation; do not proliferate aliases.

### Gates

Required gates before E2E integration:

- backend import/build proof on the mini ABI;
- packed INT8 expert weight memory ledger;
- FP4/MXFP4 -> INT8 conversion correctness against BF16 or current Marlin WNA16
  for representative experts;
- activation quantization microbench, including scale computation cost;
- grouped W13 and W2 GEMM microbench at real DSV4 token/expert shapes;
- MoE TP-boundary dtype table for
  `dsv4.v1_moe_reduce_once_all_reduce`, `dsv4.routed_expert_all_reduce`, and
  `dsv4.shared_expert_all_reduce`;
- PyNCCL/NCCL dtype microbench only if the candidate sends INT8 or INT32 through
  communication;
- no-weight replay or owner-level microbench showing expected MoE-owner win.

E2E gate:

- text smoke passes;
- logit diff/top-k stability report versus promoted exact path;
- graph replay remains zero-eager for target buckets;
- repeat-stable macro comparison against
  `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16`;
- MoE owner time improves enough to explain the macro result.

Promotion bar:

- at least `5%` MoE-owner improvement and at least `3%` E2E improvement on the
  dominant decode/serving scenarios, or a documented capacity win with no
  material throughput regression;
- quality drift is understood and accepted;
- rollback is one env/variant change.

Stop if:

- no real SM80 INT8 grouped MoE backend exists or can be compiled into mini;
- activation quantization/dequantization dominates the Tensor Core win;
- the route requires INT8 all-reduce without a scale-aware reduction design;
- quality drift is unexplained;
- the route only wins a tiny standalone GEMM but loses no-weight replay or E2E.

## Lane B: FP8 KV / MLA / Indexer Cache

### Hypothesis

DSV4 attention has several cache-like components: compressed/MLA KV state,
SWA-related state, C4/C128/indexer state, and prefix-cache ownership metadata.
FP8 cache storage may recover memory capacity and reduce bandwidth, but on SM80
it is mainly a storage/dequantization strategy, not an FP8 Tensor Core compute
strategy.  It only makes sense if the store/quant and gather/dequant boundaries
are fused enough to avoid extra HBM round trips.

### Required Source Questions

Map SGLang and vLLM before implementing:

- Which exact DSV4 KV/cache components are stored as FP8?
- Which components stay BF16, especially RoPE tail or scale metadata?
- What is the token layout, bytes/token, block/page stride, and scale format?
- Where does quantization happen:
  - fused norm/RoPE/store;
  - store-cache kernel;
  - attention gather/dequant kernel;
  - eager boundary outside kernels?
- Where does dequantization happen:
  - gather selected indices then dequant;
  - dequant full blocks then gather;
  - inside sparse attention;
  - inside a prefill-only reference path?
- How does the layout interact with radix prefix cache, Route B component
  ownership, CUDA graph buffers, and page size `256`?

Known source signals to verify:

- SGLang's DSV4 MLA cache dimension logic stores the RoPE dimension in original
  dtype while FP8 cache adds scale storage.
- SGLang has a fused `mla_kv_pack_quantize_fp8` helper and DSV4 cache-store
  kernels.
- vLLM canonicalizes DeepSeek V4 FP8 cache to `fp8_ds_mla` and has SM80
  gather/dequant fallback paths such as `gather_dequant_two_scopes_with_mask`
  and `dequantize_and_gather_k_cache`.

Do not assume the older `584 bytes/token` note applies blindly.  Recompute the
layout from current model config and current SGLang/vLLM source.

### Suggested Opt-In Surface

Prefer an explicit cache format:

```text
MINISGL_DSV4_SM80_FP8_MLA_KV_CACHE=1
MINISGL_DSV4_SM80_FP8_KV_CACHE_LAYOUT=sglang_scaled
```

If mini later gains a general `--kv-cache-dtype` surface, map these DSV4 flags
to that public API instead of keeping duplicate knobs.

### Capacity Ledger

Every FP8 KV/cache target must report:

- bytes/token per component before and after;
- GiB/rank saved at current `--num-pages 128`, page size `256`;
- equivalent additional pages and tokens;
- effect on graph capture memory headroom;
- effect on prefix-cache retained pages;
- any extra workspace needed for gather/dequant.

If the feature is mainly capacity, classify it as a capacity mode instead of
pretending it is a throughput optimization.

### Gates

Required gates before E2E integration:

- source-derived layout parity with SGLang/vLLM;
- standalone store/quant microbench;
- standalone gather/dequant microbench;
- correctness comparison against BF16 cache on selected rows/pages;
- prefix-cache hit-time materialization/remap compatibility check;
- graph capture compatibility check.

E2E gate:

- text smoke passes;
- logit diff/top-k stability report versus promoted exact path;
- prefix-cache verifier passes when prefix cache is enabled;
- graph replay remains zero-eager;
- repeat-stable macro comparison against the TARGET 10 baseline;
- capacity ledger shows a meaningful memory win.

Promotion bar:

- for throughput mode: at least `3%` repeat-stable E2E improvement without
  material correctness drift;
- for capacity mode: meaningful context/page capacity increase with no material
  throughput regression on no-hit and prefix workloads;
- rollback is one env/variant change.

Stop if:

- SM80 store/quant is forced into a slow standalone path that dominates decode;
- gather/dequant requires full-cache dequantization;
- prefix cache or graph replay breaks;
- quality drift is unexplained;
- memory saved is too small after accounting for scales/workspace.

## Lane C: Quantized Projection / Cache-Boundary Fusion

This is lower priority than INT8 MoE and FP8 KV/cache.

Only reopen this lane if a fresh post-TARGET-10 profile shows projection or
cache-boundary HBM traffic is material.

Questions:

- Can quantization be moved inside GEMM or cache-store kernels to avoid extra
  HBM reads/writes?
- Are there vLLM/SGLang kernels worth porting into mini's ABI?
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
- whether weights, activations, KV cache, indexer cache, logits, or
  communication dtype changed;
- source of scales and quantization algorithm;
- whether casts happen inside fused kernels or as standalone HBM round trips;
- SGLang/vLLM parity status;
- text smoke result;
- logit diff or top-k stability where feasible;
- graph replay/eager decode status;
- memory capacity delta;
- TP8 macro result with repeat-stable comparison;
- rollback command.

Suggested correctness tiers:

- Tier 0: kernel-level numerical check against BF16/exact reference.
- Tier 1: text smoke with generated outputs saved.
- Tier 2: logit diff/top-k stability over fixed prompt sets.
- Tier 3: serving-style smoke with prefix cache enabled when cache state
  changes.

## Promotion Rules

Promote a low-precision feature only if all are true:

- correctness/quality gates pass;
- no silent fallback;
- graph replay remains active;
- repeat-stable E2E gain is at least `3%`, or the feature is explicitly marked
  as a memory/capacity mode;
- memory/workspace accounting is documented;
- a rollback env/variant exists.

Precision-risk features should remain opt-in until they have broader quality
coverage than the engineering smoke gates above.

## Stop Rules

Stop a lane if:

- the required backend does not support SM80 and cannot be built into mini's ABI
  within the target;
- standalone quant/dequant overhead dominates the expected win;
- quality drift is unexplained;
- prefix cache compatibility is broken;
- the macro result is neutral or negative after a fair repeat-stable gate;
- the thread starts optimizing a custom local implementation while a SGLang or
  vLLM backend has not yet been mapped.

## Non-Goals

- Replacing TARGET 08 prefix cache.
- Changing the promoted exact-ish bundle by default.
- Changing communication routing from the TARGET 10 default.
- Running INT8 MoE and FP8 KV/cache in one child thread.
- Running broad quality evaluation beyond the smoke/oracle gates needed for an
  engineering decision.
- Porting large vLLM/SGLang subsystems wholesale without isolating the actual
  backend boundary that provides the win.

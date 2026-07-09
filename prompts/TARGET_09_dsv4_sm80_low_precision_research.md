# TARGET 09: DSV4 SM80 Low-Precision Research Roadmap

## Status

Deferred for now.

The fine-grained TARGET 09 prompts are archived under:

```text
prompts/archive/target09/
```

Use this root file as the summary.  Reopen an archived prompt only when a fresh
profile or capacity target makes that lane clearly valuable again.

## Why This Target Exists

TARGET 09 is intentionally separate from the exact TARGET 07/08/10 route
because it may change activation, cache, expert, or communication precision.
Those changes can affect correctness and output quality, so they need stronger
quality gates than exact BF16/FP4-weight paths.

The two original lanes were:

1. INT8 MoE W8A8: convert model-native FP4 expert weights to an INT8
   backend-specific layout and quantize activations at the MoE boundary so
   routed experts can use A100 INT8 Tensor Cores.
2. FP8 KV/cache: align with SGLang/vLLM DeepSeek V4 cache layouts and decide
   whether selected cache components should be stored in FP8.

The guiding rule remains:

```text
Use SGLang/vLLM as implementation oracles first.  If mini uses a different
algorithm, prove why it is simpler, correct, and faster before integrating it.
```

## Current Decision

TARGET 09 remains deferred.  TARGET 11 MTP speculative decoding was explored
after the first low-precision pass and is now paused for release, so it is no
longer the active reason to keep TARGET 09 closed.

The most recent low-precision evidence did not show an obvious short win:

- The exact MoE path is already strong because model-native MXFP4/WNA16 Marlin
  is the winning A100/sm80 expert backend.
- INT8 MoE remains possible in theory, but the standalone/backend feasibility
  work did not yet identify a low-risk path that clearly beats WNA16 after
  activation quantization, layout conversion, output dequant, and TP reduce
  costs.
- FP8 KV/cache capacity ROI became less compelling after TARGET 08 separated
  SWA lifecycle and recovered capacity with Marlin WNA16 release.
- Dense FP8 Marlin projection was speed-neutral on the promoted path and should
  be treated as a memory/capacity feature unless a later bottleneck changes.
- Naive FP8 indexer/logits or standalone quant/dequant insertion can easily
  lose more time than it saves.

## Baseline For Any Future Reopen

Use the latest promoted exact/prefix/communication route, not an old TARGET 07
baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL default threshold32m
--page-size 256
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

If TARGET 08 has promoted newer SWA/Marlin release presets after this file was
written, use the newer promoted capacity baseline.

## Archived Split Plan

Historical child prompts:

| Stage | Archived Prompt | Decision |
| --- | --- | --- |
| TARGET 09.0 | `prompts/archive/target09/TARGET_09.0_dsv4_sm80_low_precision_preflight.md` | Completed preflight/census route. |
| TARGET 09.1 | `prompts/archive/target09/TARGET_09.1_dsv4_sm80_int8_moe_backend_feasibility.md` | INT8 MoE remains research; no obvious short integration win yet. |
| TARGET 09.2 | `prompts/archive/target09/TARGET_09.2_dsv4_sm80_int8_moe_optin_integration.md` | Conditional only if a future backend beats WNA16 in microbench. |
| TARGET 09.25 | `prompts/archive/target09/TARGET_09.25_dsv4_sm80_int8_comm_boundary_feasibility.md` | Optional research; do not mix into INT8 MoE compute unless scale semantics and speed are proven. |
| TARGET 09.3 | `prompts/archive/target09/TARGET_09.3_dsv4_sm80_fp8_kv_cache_parity_ledger.md` | Source/capacity ledger route. |
| TARGET 09.4 | `prompts/archive/target09/TARGET_09.4_dsv4_sm80_minimal_fp8_kv_cache_slice.md` | Minimal FP8 cache slice only if capacity ROI becomes compelling. |
| TARGET 09.45 | `prompts/archive/target09/TARGET_09.45_dsv4_sm80_fp8_cache_roi_sglang_lifecycle.md` | Selected lifecycle/capacity-first before FP8 cache E2E. |
| TARGET 09.5 | `prompts/archive/target09/TARGET_09.5_dsv4_sm80_fp8_kv_cache_optin_e2e.md` | Deferred. |
| TARGET 09.6 | `prompts/archive/target09/TARGET_09.6_dsv4_sm80_quantized_projection_cache_boundary_fusion.md` | Optional later if projection/cache traffic is top bottleneck. |

## If Reopened Later

Do not implement INT8 MoE and FP8 KV/cache in the same child thread.  They have
different correctness, memory, graph-capture, and quality failure modes.

Reopen INT8 MoE only if:

- a real SM80 grouped MoE W8A8 backend exists or can be compiled into mini's ABI;
- FP4/MXFP4 expert weights can be converted once on load into the backend
  layout;
- activation quantization is fused or cheap enough not to erase the Tensor Core
  win;
- output can return to BF16 before the promoted TP reduce path unless a separate
  INT8 communication target proves safe scale semantics;
- microbench beats WNA16 by enough margin to justify E2E work.

Reopen FP8 KV/cache only if:

- a fresh memory ledger shows meaningful capacity pressure after current
  SWA/Marlin release behavior;
- the target component matches SGLang/vLLM layout and cast placement;
- store/quant and gather/dequant kernels are proven on SM80;
- quality smoke and prefix/SWA ownership gates are planned from the start.

## Current Recommendation

Keep TARGET 09 deferred until a fresh profile or memory ledger shows that
low-precision work is the best next lever.  After MTP cleanup, first establish a
non-MTP post-cleanup speed and correctness baseline on the promoted TARGET 10
path.

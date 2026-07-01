# vLLM FusedMoE Runner Integration Sketch

Date: 2026-07-01

This is the concrete follow-up plan from TARGET 07.35. It is not an
implementation artifact. The intended next cut is to adapt vLLM's FusedMoE
execution shape into mini's exact DeepSeek V4 path without making vLLM a
runtime dependency and without changing mini's default precision policy.

## What To Adapt

Use vLLM as a design reference for these boundaries:

- `DeepseekV4MoE` creates a standard `FusedMoE` path on sm80; do not port
  `DeepseekV4MegaMoEExperts`.
- `MoERunner.forward/_forward_impl` owns the route, shared expert scheduling,
  dispatch/combine, routed output transform, and final reduce decision.
- `FusedMoEPrepareAndFinalizeModular` defines `prepare` and `finalize`.
- `FusedMoEExpertsModular` defines `workspace_shapes`, `workspace_dtype`,
  expert compute, and the top-k weight/reduce semantic.
- `SharedExperts` owns the stream ordering choice: no overlap,
  MK-internal overlap, or multi-stream overlap.
- `moe_align_block_size` route metadata uses `sorted_token_ids`,
  `expert_ids`, and `num_tokens_post_padded`.

## What Not To Adapt In This Exact Target

- No activation quantization.
- No INT8 MoE.
- No vLLM runtime dependency.
- No default MXFP4/FP8 precision semantic change. Keep this for TARGET 07.4.
- No `DeepseekV4MegaMoEExperts` on sm80.
- No FlashInfer/TRTLLM dependency as part of the exact runner boundary.

## Mini Runner Cut

Add a mini-owned exact runner, for example `DSV4FusedMoERunner`, with these
subobjects:

| Component | Mini responsibility | vLLM reference |
| --- | --- | --- |
| Router | Reuse current `DSV4MoEGate`; return fp32 top-k weights and int top-k ids. | `FusedMoERouter.select_experts`, `fused_topk_bias`. |
| Prepare/finalize | Standard no-DP/EP path first: no activation quantization, no all-to-all, output not reduced. Finalize applies route weights/top-k sum into `[M, K]`. | `MoEPrepareAndFinalizeNoDPEPModular`. |
| Experts | Wrap existing exact grouped FP4 W13/SwiGLU/W2 kernels behind `apply`. Expose vLLM-like `workspace13`, `workspace2`, and output sizing. | `FusedMoEExpertsModular`, `CutlassExpertsFp4.workspace_shapes`. |
| Shared experts | Move shared expert scheduling into the runner. First pass can be serial; second pass can add opt-in aux-stream overlap under a token threshold. | `SharedExperts`, `SharedExpertsOrder`. |
| Reduce boundary | Keep one late TP all-reduce after routed + shared local sum. | `MoERunner._maybe_reduce_final_output`. |
| Workspace | Use per-layer reusable workspace with explicit max-size policy. Keep prefill guard to avoid retaining huge routed buffers. | `FusedMoEKernelModularImpl._allocate_buffers`. |

The first implementation should route `DSV4MoE.forward` through this runner
while preserving the exact old output bit/tolerance behavior. The old V1/V2
paths should remain as fallbacks until the runner passes microbench, text smoke,
and macro gates.

## Measurement Gates

Correctness:

- Unit test runner output vs current `DSV4MoE` for routed-only, shared-only,
  routed+shared, hash routing, and correction-bias routing.
- TP8 text smoke:
  `benchmark/offline/deepseek_v4_text_smoke.py --variants <runner-variant>`.
- No new `MINISGL_DSV4_SM80_MOE_INT8`, activation quantization, MXFP4/FP8 cache,
  or vLLM dependency.

Microbench:

- Extend `benchmark/offline/deepseek_v4_moe_route_microbench.py` with:
  `runner_prepare_ms`, `runner_experts_ms`, `runner_finalize_ms`,
  `runner_shared_ms`, and `runner_total_ms`.
- Compare against current V1/V2 grouped full timings and require exact or
  existing-tolerance output agreement.

Macro:

- Run 4096/128/batch4 TP8 for a short profile-equivalent artifact.
- Run 4096/1024/batch4 TP8 as the official macro.
- Record graph replay coverage, communication labels, fallback wrapper totals,
  and decode forward time.

Profile:

Capture Nsight on 4096/128/batch4 after the runner cut:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=performance_milestones/target07_vllm_fused_moe_runner/raw/nsys_runner_4096x128_bs4 \
  torchrun --standalone --nproc_per_node=8 \
    benchmark/offline/deepseek_v4_perf_matrix.py \
      --variants <runner-variant> \
      --scenarios mixed_prefill_decode_bs4 \
      --prompt-len 4096 \
      --decode-len 128 \
      --batch-size 4 \
      --repeats 1 \
      --warmup-repeats 0 \
      --output-dir performance_milestones/target07_vllm_fused_moe_runner/raw/mini_4096_128_bs4_runner_perf
```

## Stop Rule For The Runner Cut

Stop MoE work and re-rank again if the runner-boundary cut gives less than 5
percent 4096/1024 macro gain and less than 10 percent routed-MoE subgraph gain,
or if a fresh Nsight artifact shows attention/cache/indexer ahead of MoE by a
clear margin. In that case, open the attention/cache/indexer target next.

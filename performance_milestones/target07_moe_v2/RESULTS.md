# TARGET 07.3 MoE V2 Results

Date: 2026-07-01
GPU: 8x NVIDIA A100-SXM4-80GB

## Correctness

- Targeted pytest: 54 passed, 4 external deprecation warnings.
- Microbench V2 vs V1 grouped max abs: 0.0 for all default cases.

## Microbench Smoke

Artifact: `moe_v2_microbench_smoke.json`

| case | V1 full ms | V2 full ms | V2 dispatch ms | V2 vs V1 max abs |
| --- | ---: | ---: | ---: | ---: |
| decode_tiny | 0.2822 | 0.2976 | 0.1346 | 0.0 |
| decode_grouped | 0.3037 | 0.3277 | 0.2599 | 0.0 |
| prefill_grouped | 0.4411 | 0.4458 | 0.4006 | 0.0 |

Interpretation: the execution-plan/workspace dispatch path is correct and can
reduce dispatch-only time on tiny decode-like shapes, but plan construction plus
dispatch is not yet a net win.

## Macro Smoke

Scenario: `mixed_prefill_decode_bs4` with `prompt_len=4096`, `decode_len=1024`,
`batch_size=4`, `repeats=1`, `warmup_repeats=0`, TP8.

Artifacts:

- V1 baseline: `macro_4096_1024_bs4_tp8_smoke/`
- V2 after workspace guard: `macro_4096_1024_bs4_tp8_v2_after_workspace_guard/`
- V2 after bf16-output SwiGLU: `macro_4096_1024_bs4_tp8_v2_bf16_swiglu/`

| variant | status | decode tok/s | e2e output tok/s | prefill tok/s | TTFT mean s |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 current best | pass | 19.9058 | 17.3596 | 611.6472 | 17.2379 |
| V2 first cut | pass | 19.9214 | 17.8116 | 783.9889 | 13.5572 |
| V2 bf16-output SwiGLU | pass | 19.9037 | 17.8009 | 782.6677 | 13.5613 |

Ratios:

- V2 first cut vs V1 decode tok/s: 1.0008x
- V2 first cut vs V1 end-to-end output tok/s: 1.0260x
- V2 bf16-output SwiGLU vs V2 first cut decode tok/s: 0.9991x
- V2 bf16-output SwiGLU vs V1 decode tok/s: 0.9999x

## Cut Notes

Cut 1 added the explicit V2 route execution plan and per-layer workspace. The
first unguarded macro run showed the useful failure mode: persistent per-layer
workspace retained prefill-sized routed buffers and caused CUDA OOM. The
implementation now only uses reusable workspace for decode-sized route counts;
large prefill uses ephemeral temporaries.

Cut 2 added a bf16-output SwiGLU wrapper for materialized grouped MoE, replacing
`SwiGLU fp32 output + torch cast to bf16` with one Triton store into bf16. It is
correct, but did not improve macro or default routed-MoE microbench timings.

## Stop-Rule Read

Stop 07.3 now. Two consecutive MoE cuts were below the stop thresholds:

- Cut 1 macro decode gain was 1.0008x, below 5 percent and far below 1.3x.
- Cut 2 macro decode ratio versus cut 1 was 0.9991x, below 5 percent.
- Default routed-MoE microbench did not show a 10 percent subgraph gain.
- No W13/W2 summed kernel-time 2x result has been demonstrated.

Per `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`, the next target should be
`prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md`, not another 07.3 MoE cut.

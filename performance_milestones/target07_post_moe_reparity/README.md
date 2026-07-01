# TARGET 07.35: DSV4 sm80 Post-MoE Re-Parity

Date: 2026-07-01

Status: complete. TARGET 07.3 hit its stop condition after two exact MoE V2
cuts, so this milestone freezes the post-MoE baseline, re-ranks the remaining
gap, and chooses the next focused plan.

## Artifacts

| Path | Contents |
| --- | --- |
| `summaries/post_moe_metrics.json` | Small machine-readable summary of correctness, macro, short profile-equivalent, and MoE microbench results. |
| `summaries/text_smoke_v2_exact.json` | TP8 text correctness smoke for the current exact V2 macro variant. |
| `summaries/vllm_fused_moe_runner_integration.md` | Concrete next-cut plan for adapting vLLM's FusedMoE runner shape into mini. |
| `raw/mini_4096_128_bs4_v2_exact_perf/` | Fresh 4096/128/batch4 TP8 profile-equivalent perf matrix run. |
| `raw/mini_4096_1024_bs4_v2_exact_perf` | Symlink to the TARGET 07.3 4096/1024/batch4 exact V2 macro artifact. |
| `raw/mini_4096_1024_bs4_v1_baseline_perf` | Symlink to the TARGET 07.3 V1 current-baseline macro artifact. |
| `raw/moe_v2_microbench_smoke.json` | Symlink to the TARGET 07.3 MoE V2 microbench artifact. |

## Frozen Mini Baseline

Variant:
`v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`

Environment policy:

- model path: `/models/DeepSeek-V4-Flash`
- TP size: 8
- page size: 256
- dtype: bf16
- communication backend: NCCL
- decode CUDA graph sizes: `[1, 2, 4]`
- greedy sampler captured in graph: yes
- exact target lane: inherited TARGET 07.2/07.3 exact macro variant with V1
  grouped FP4 MoE plus V2 route execution plan/workspace
- not introduced in 07.35: new activation quantization, INT8 MoE, vLLM
  runtime dependency, or MXFP4/FP8 precision-lane promotion

Git state in the fresh 4096/128 run:

- branch: `dsv4-sglang-based`
- commit: `3871626`
- dirty: yes, containing the TARGET 07.3 MoE V2 implementation and these new
  milestone artifacts

Correctness:

- TARGET 07.3 targeted pytest: 54 passed, 4 external warnings.
- 07.35 TP8 text smoke: pass.
- Text smoke graph replay: 9/9 decode replays, no eager decode fallback.

## Macro Results

Scenario: `mixed_prefill_decode_bs4`, prompt len 4096, batch size 4, TP8,
repeats 1, warmup repeats 0.

| Workload | Status | Decode tok/s | E2E output tok/s | Prefill tok/s | TTFT mean s | Decode forward s | Comm count | Comm bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/1024 exact V2 | pass | 19.9037 | 17.8009 | 782.6677 | 13.5613 | 128.4185 | 704 | 87.26 GB |
| 4096/128 exact V2 | pass | 19.8442 | 10.7751 | 784.9714 | 13.5283 | 15.9240 | 704 | 87.26 GB |

4096/128 graph replay coverage:

- 127 decode graph replays
- 0 eager decode fallback
- padded replay sizes: 32 at 1, 32 at 2, 63 at 4

Reference lines:

| Reference | 4096/1024 output tok/s |
| --- | ---: |
| old vLLM serving victory line | 114.07 |
| fair vLLM offline reference from TARGET 07.25 | 201.8738 |
| mini TARGET 07.2 best exact reference | about 25.3076 |
| current post-MoE exact V2 fresh baseline | 17.8009 |

This does not reach the old serving victory line. It is also not a macro win
over the TARGET 07.2 best exact reference. The current V2 fresh result is
therefore a baseline for the next cut, not a promoted performance win.

## MoE V2 Read

TARGET 07.3 microbench smoke:

| Case | V1 grouped full ms | V2 grouped full ms | V2 dispatch ms | V2 vs V1 max abs |
| --- | ---: | ---: | ---: | ---: |
| decode_tiny | 0.2822 | 0.2976 | 0.1346 | 0.0 |
| decode_grouped | 0.3037 | 0.3277 | 0.2599 | 0.0 |
| prefill_grouped | 0.4411 | 0.4458 | 0.4006 | 0.0 |

The V2 route execution plan is correct, but it does not reduce full routed MoE
time. The second cut, bf16-output SwiGLU, was also correct but did not improve
macro. Per the TARGET 07.3 stop rule, do not keep drilling route/workspace or
single-kernel cast cleanups inside 07.3.

## Updated Bottleneck Ranking

The fresh 4096/128 and 4096/1024 perf-matrix labels are:

1. decode dominated
2. attention
3. MoE / expert GEMM
4. fp4 expert handling
5. KV cache writes
6. metadata construction

This fresh run is a profile-equivalent benchmark report, not a new Nsight
kernel-time table. Therefore it updates the priority read but does not fully
replace the TARGET 07.25 Nsight evidence, where routed MoE W13/W2 was the
largest measured kernel-time category. The post-MoE ranking is:

1. MoE runner/execution boundary and sparse attention/cache are co-top risks.
   MoE remains top-two because 07.3 did not shrink W13/W2 or the routed expert
   boundary, but attention is now equally visible in the fresh benchmark labels.
2. Sparse attention, indexer, and cache layout. This is the immediate next
   non-MoE target if a true runner-boundary cut does not move macro.
3. Scheduling, graph, and multi-stream overlap. This includes vLLM's shared
   expert aux-stream scheduling and attention aux-stream topology.
4. Communication and reduce boundary. Still 704 labeled collectives and
   87.26 GB per macro run, but not the top measured kernel-time item in the
   prior Nsight evidence.
5. Precision lane. vLLM's MXFP4/FP8 policy remains a major possible source of
   absolute parity, but it is deferred to TARGET 07.4.
6. HC/RMSNorm/final/sampling. These are not the current limiting items.

## vLLM MoE Integration Decision

Do not continue TARGET 07.3 with peripheral MoE cuts. The next MoE work, if
accepted, should be a single focused vLLM-style FusedMoE runner-boundary cut.
The design target is documented in
`summaries/vllm_fused_moe_runner_integration.md`.

The core difference is that vLLM's sm80 DeepSeek V4 path uses standard
`FusedMoE`, not `DeepseekV4MegaMoEExperts`. Its runner owns:

- route selection and route metadata
- `prepare` and `finalize`
- `workspace13`, `workspace2`, and output workspace sizing
- expert compute behind a modular experts interface
- shared expert scheduling
- late final reduce after routed plus shared output, unless the kernel already
  reduced the routed output

Mini currently has route metadata and a reusable grouped-MoE workspace, but the
runner boundary is still split across `DSV4MoE.forward`,
`DSV4FusedRoutedExperts.forward`, `DSV4SharedExperts.forward`, and the final
reduce. That is why the V2 route/workspace cut was correct yet not a macro win.

Keep the mini default exact:

- adapt the FusedMoE runner shape
- preserve the current exact-lane precision semantics and do not add a new
  activation-quantization lane
- keep current exact FP4 expert handling unless the new runner exposes a
  measured exact kernel replacement
- defer vLLM MXFP4/FP8 precision semantics to TARGET 07.4
- reject sm80 `DeepseekV4MegaMoEExperts`
- do not add vLLM as a runtime dependency

## Next Target

Selected next target: continue MoE only as one scoped runner-boundary cut,
preferably as a new milestone such as
`performance_milestones/target07_vllm_fused_moe_runner/`. If the project wants
to keep the prompt taxonomy strict, this is the one allowed MoE continuation
from 07.35; it is not permission to keep doing route/workspace/graph cleanup
inside 07.3.

First implementation cut:

1. Add a mini-owned `DSV4FusedMoERunner` abstraction.
2. Mirror vLLM's standard no-DP/EP modular path:
   `router -> prepare -> experts -> finalize -> shared+routed sum -> late reduce`.
3. Start by wrapping the existing exact grouped FP4 expert kernels, not by
   importing vLLM or changing precision.
4. Move shared expert scheduling into the runner; keep serial first, then add
   an opt-in aux-stream threshold if correctness and profile evidence support it.
5. Measure with text smoke, MoE route microbench, 4096/128 profile-equivalent,
   and 4096/1024 macro before any second cut.

Stop immediately after that runner-boundary cut if macro gain is below 5
percent and routed-MoE subgraph gain is below 10 percent, or if fresh Nsight
shows attention/cache/indexer clearly ahead of MoE. In that case, open the
attention/cache/indexer target next.

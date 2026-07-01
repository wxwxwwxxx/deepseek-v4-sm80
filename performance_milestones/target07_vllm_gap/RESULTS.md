# TARGET 07.1 Results: DSV4 SM80 vLLM Gap

Status: complete enough to enter TARGET 07.2.

Scope: DeepSeek V4 Flash, A100/sm80, TP8, page/block size 256,
4096 prompt tokens, batch size 4, one warmup repeat, one measured repeat.

## Macro 4096/1024 Result

| System | Shape | Output tok/s | Elapsed s | Notes |
| --- | --- | ---: | ---: | --- |
| mini v1_moe fair | 4096/1024/bs4 | 10.5768 | 387.264 | TTFT 21.944 s, decode 11.261 tok/s |
| vLLM fair | 4096/1024/bs4 | 201.874 | 20.290 | chunked prefill 4096, CUDA graph sizes 1/2/4 |
| old vLLM serving baseline | 4096/1024/bs4 | 114.07 | n/a | TTFT 123.21 ms, TPOT 15.68 ms |

Ratios:

- vLLM fair / mini fair: 19.09x.
- old baseline / mini fair: 10.78x.
- vLLM fair / old baseline: 1.77x.

Conclusion: the old 114.07 output tok/s baseline is not an unreachable target.
The fair vLLM run is substantially faster than that baseline. The mini gap is
therefore an execution-path gap inside mini, not mainly a benchmark fairness
artifact.

## Nsight 4096/128 Result

| System | Shape | Output tok/s | Elapsed s | TTFT s | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| mini default prefill | 4096/128/bs4 | 5.5071 | 92.972 | 22.623 | rank0 nsys, `NSYS_MEMORY_RATIO=0.8` |
| mini `MAX_EXTEND_TOKENS=4096` | 4096/128/bs4 | 5.3471 | 95.752 | 15.612 | rank0 nsys |
| vLLM fair | 4096/128/bs4 | 80.9050 | 6.328 | n/a | full vLLM profile |

Ratios:

- vLLM 4096/128 / mini default 4096/128: 14.69x.
- mini max-extend / mini default: 0.971x.

Conclusion: matching vLLM's chunked prefill shape does not explain the gap.
`MAX_EXTEND_TOKENS=4096` lowers mini TTFT in this short profile, but it slightly
hurts end-to-end output throughput and does not move decode speed meaningfully.

## Nsight Execution Shape

| Metric | mini default rank0 | mini max-extend rank0 | vLLM total |
| --- | ---: | ---: | ---: |
| CUDA kernels | 6,663,421 | 6,824,955 | 124,480 |
| CUDA kernel duration | 91.154 s | 95.258 s | 0.979 s |
| CUDA runtime calls | 7,389,232 | 7,563,623 | 1,908,662 |
| CUDA runtime duration | 94.533 s | 99.774 s | 46.598 s |
| NCCL kernels | 22,528 | 23,056 | 16 |
| NCCL kernel duration | 1.481 s | 1.467 s | 0.021 s |
| CUDA graph events | 0 | 0 | 7,200 |

mini default versus vLLM:

- 53.53x more CUDA kernels.
- 3.87x more CUDA runtime calls.
- 1408x more NCCL kernels.
- No DSV4 CUDA graph replay in mini; vLLM shows graph creation/execution events.

Top mini default kernels by GPU time:

| Kernel | Count | Duration s |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11,008 | 28.361 |
| `_grouped_fp4_linear_kernel` | 11,008 | 18.713 |
| `sparse_attention_kernel` | 10,496 | 8.086 |
| PyTorch direct copy elementwise | 532,968 | 6.137 |
| PyTorch div elementwise | 952,064 | 2.597 |

Profiler caveat: mini nsys uses rank0-only profiling because wrapping the whole
`torchrun` launcher triggered a container/Nsight/NCCL crash. vLLM's decode NVTX
window has poor kernel attribution, so the vLLM comparison above uses total
profile counts. The direction is still stable: mini has far more launches, far
more collective kernels, and no CUDA graph replay.

Artifact note: the successful mini default-prefill nsys run used
`NSYS_MEMORY_RATIO=0.8` for profiler headroom and has been promoted to the
official default-prefill artifact names. The older ad-hoc names are kept only as
backward symlinks outside this milestone directory.

## Diagnosis

The main TARGET 07 gap is now pinned to runtime execution structure:

1. Communication is too fragmented. mini emits 22,528 NCCL kernels on one
   profiled rank for the short run, while vLLM emits 16 total NCCL kernels.
2. DSV4 decode CUDA graph is absent in mini. The code path still forces DSV4
   graph sizes to empty, so every decode step launches eagerly.
3. MoE expert compute remains expensive. The two grouped FP4 kernels dominate
   mini GPU time, but fixing MoE first would still leave millions of small
   launches and thousands of collectives.
4. PyTorch small-kernel overhead is large. The profile shows hundreds of
   thousands of elementwise/copy/reduction launches.
5. Chunked prefill fairness is not the blocker. The max-extend mini run did not
   close the gap.

## vLLM Design Decisions

The execution-path decisions are recorded in `EXECUTION_DIFF.md`:

- `port`: preserve and enforce late MoE all-reduce.
- `adapt`: communication custom-op boundary, PyNCCL/custom all-reduce ideas,
  CUDA graph dispatcher shape, DSV4 attention custom-op boundary, shared expert
  overlap structure, sparse decode layout ideas.
- `reject`: vLLM SM80 reference sparse prefill as a default path, because it has
  already shown OOM risk for this work.
- `defer`: activation quantization, FP8/FP4 activation lanes, INT8/TF32
  experiments, and full backend-policy parity.

## Next Path

Enter TARGET 07.2 now. Do not spend another cycle on TARGET 07.1 unless a
specific disputed number needs confirmation.

Recommended TARGET 07.2 order:

1. Add per-site communication labels/counters and keep the late MoE reduce as a
   named invariant.
2. Run an exact-path PyNCCL comparison for 4096/128 and 4096/1024. If correct
   and faster, make it the DSV4 TP8 benchmark path or guarded default.
3. Prepare collectives and attention/MoE boundaries for CUDA graph capture.
4. Enable DSV4 decode CUDA graph for capture sizes `1,2,4` only after metadata
   replay is stable; verify exactness before treating it as default.
5. Reprofile. If grouped FP4 MoE still dominates after communication and graph
   replay, then move to TARGET 07.3 MoE V2.

Do not make vLLM a runtime dependency. Do not port vLLM's OOM-prone SM80 sparse
prefill path. Keep the first optimization phase on the bf16-direct exact path.

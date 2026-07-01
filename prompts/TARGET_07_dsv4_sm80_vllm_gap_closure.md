# TARGET 07: DeepSeek V4 sm80 vLLM Gap Closure

## Goal

Close the remaining DeepSeek V4 Flash performance gap between mini-sglang and
the old vLLM-based framework on A100/sm80.

The primary win condition is:

- TP8, single-node 8x A100 sm80
- page/block size 256
- `/models/DeepSeek-V4-Flash`
- 4096 input tokens/request
- 1024 output tokens/request
- batch size 4
- output throughput strictly above the old vLLM serving baseline:
  `114.07 tok/s`

The default promoted path must remain exact. Approximate INT8 Tensor Core MoE
is allowed only as an opt-in research variant until it passes explicit quality
gates.

## Precision Policy

mini-sglang does not need to match the old vLLM framework's precision policy in
the first optimization stage. The goal is to beat vLLM performance while keeping
mini's intended accuracy policy explicit and measurable.

Precision roadmap:

1. `bf16-direct` first.
   - Major sm80 kernels should prefer bf16 Tensor Core paths.
   - Do not add activation quantization to the default path.
   - Preserve model-original fp32 computations where they exist. Do not silently
     downcast fp32 model math to bf16.
   - TF32 may be tested for fp32 matmul-like work only as an explicit
     experiment, with correctness/text-smoke evidence before keeping it.

2. fp8/fp4 activation quantization experiments second.
   - Add fp8-act or fp4-act only as opt-in variants after the bf16-direct path
     is measured.
   - Use vLLM's implementation as a priority reference when its precision lane
     matches the experiment.
   - Decide whether activation quantization is worth the accuracy and
     complexity cost using microbench, E2E, logits/top-k, and text-smoke data.

3. INT8 Tensor Core path third.
   - Treat INT8 as a higher-risk opt-in lane, especially for MoE expert compute.
   - Compare against the best exact bf16-direct path, not against a weak
     fallback.
   - Do not promote INT8 unless both performance and quality gates pass.

## Current Evidence

V1 MoE proved the right first bottleneck: it removed the worst FP4 expert
fallback loop and improved local E2E decode workloads by about 4.5x-7.8x.
However, the old vLLM-based framework is still about an order of magnitude
faster on the 4096/1024/batch4 workload.

Important recorded artifacts:

- mini V1 milestone: `performance_milestones/v1_moe/README.md`
- vLLM comparison scripts/artifacts: `performance_milestones/vllm/README.md`
- TARGET 07.2 communication/graph trajectory:
  `performance_milestones/target07_comm_graph/README.md`
- mini short nsys: `performance_milestones/v1_moe/raw/nsys_mini_v1_moe_4096x128_bs4.sqlite`
- vLLM short nsys: `performance_milestones/vllm/raw/nsys_vllm_4096x128_bs4.sqlite`

Short 4096/128/batch4 nsys facts:

| Metric | mini V1 MoE | old vLLM |
| --- | ---: | ---: |
| E2E output throughput | 5.07 tok/s | 80.83 tok/s |
| elapsed | 101.08s | 6.33s |
| CUDA kernel events | ~26.8M | ~124K |
| CUDA runtime calls | ~30.5M | ~1.9M |
| mini fallback wrapper calls | ~1.43M | n/a |

mini V1 top kernel-time categories in the formal workload window:

- NCCL all-reduce bf16/f32: about 262s summed GPU time
- grouped MoE FP4 W13/W2: about 188s summed GPU time
- PyTorch elementwise/copy/reduce small kernels: about 114s summed GPU time
- sparse attention: about 33s summed GPU time
- indexer bf16 logits: about 8s summed GPU time

Interpretation: the next gap is not just one kernel. It is execution shape:
too many collectives, too many small kernels, no DSV4 CUDA graph replay, and a
still-not-final grouped MoE path.

TARGET 07.2 update: communication counters, PyNCCL fp32 support, guarded DSV4
decode CUDA graph, graph-body HC/RMSNorm helpers, and several vLLM-aligned
attention-boundary cleanups improved the best exact 4096/1024/batch4 result to
about 25.3 output tok/s. This is a useful 2.4x improvement over the fair mini
V1 baseline, but still far below the fair vLLM macro result recorded in that
milestone. Further small graph-surface changes produced tiny gains. The plan
therefore pivots to subgraph-level mini-vs-vLLM parity analysis before more
implementation-heavy optimization.

TARGET 07.25 update: the subgraph parity milestone is recorded in
`performance_milestones/target07_subgraph_parity/README.md`. The remaining
4096/1024/batch4 gap is now ranked as: (1) MoE routed experts and MoE execution
boundary, (2) sparse attention/indexer/cache layout, (3) scheduling/graph and
multi-stream overlap, (4) communication/reduce boundary, (5) vLLM-only
precision lane, and (6) HC/RMSNorm/final/sampling. The next implementation
target is TARGET 07.3 MoE exact V2. vLLM's FusedMoE runner shape should be
adapted, while MXFP4/FP8 activation or KV/cache precision should remain a
separate deferred precision-lane target.

TARGET 07.35 update: TARGET 07.3's exact MoE V2 route-plan/workspace and
bf16-output SwiGLU cuts were correct, but they did not improve decode
throughput. The post-MoE re-parity milestone is recorded in
`performance_milestones/target07_post_moe_reparity/README.md`. The current
exact V2 4096/1024/batch4 smoke macro is about `17.8` E2E output tok/s and
`19.9` decode tok/s under the recorded policy, with TP8 page-size-256 text smoke
passing. The next implementation target is no longer peripheral 07.3 MoE
cleanup. It is TARGET 07.36: adapt vLLM's standard FusedMoE runner boundary into
mini as a mini-owned exact baseline, then decide from fresh data whether to keep
MoE work, open attention/cache/indexer, or move to precision/backend lanes.

TARGET 07.36 update: the vLLM-shaped mini-owned FusedMoE runner is implemented
and measured in `performance_milestones/target07_vllm_fused_moe_runner/`. It
passes correctness and TP8 text smoke, but 4096/1024/batch4 improves only about
`+0.16%`, and DSV4-like routed-MoE microbench is neutral-to-negative. Fresh
4096/128 Nsight still shows grouped FP4 W13/W2 as dominant, about `46.781s` plus
`31.700s` summed GPU time. Therefore stop MoE wrapper/runner work. The next
step is split into TARGET 07.37 backend identification first, and TARGET 07.38
exact backend adaptation only if 07.37 proves a vLLM-style exact W4A16 backend
is feasible and worth porting.

TARGET 07.38 update: TARGET 07.37 selected Marlin MXFP4 W4A16 as the actual
vLLM SM80 MoE expert backend, but direct adaptation stopped with a precise
blocker in `performance_milestones/target07_moe_exact_backend_adapt/`: mini
does not own the required Marlin custom-op surface (`gptq_marlin_repack` and
`_moe_C::moe_wna16_marlin_gemm`). The default grouped FP4 backend remains
unchanged and the Marlin opt-in guard fails explicitly instead of silently
falling back. TARGET 07.39 is the next evidence target: use the locally
installed vLLM compiled ops only as an experimental bridge/probe to determine
whether a narrow mini-owned Marlin csrc port is worth opening.

TARGET 07.39 update: the bridge probe in
`performance_milestones/target07_marlin_custom_op_bridge/` can import and call
the locally installed vLLM Marlin custom ops on A100 SM80. Synthetic DSV4-like
T=4 and T=4096 MoE calls pass, route metadata is semantically compatible, and
fused Marlin is about `4.55x` and `20.05x` faster than mini grouped FP4 in the
probe. The bridge remains external/probe-only; the explicit
`vllm_marlin_bridge` marker fails at mini runtime instead of depending on vLLM
or silently falling back. Next step: open a narrow mini-owned Marlin WNA16 csrc
port target, not TARGET 07.4.

TARGET 07.391 update: the mini-owned csrc port in
`performance_milestones/target07_marlin_wna16_csrc_port/` is implemented and
model-integrated as explicit backend `marlin_wna16`. It vendors the narrow
Marlin WNA16 source surface, builds/loads without a vLLM runtime dependency,
transforms and caches MXFP4 expert weights, passes TP8 text smoke, and runs
4096/128 plus 4096/1024 batch4 macro with CUDA graph replay and zero unsupported
skips when `--num-pages 128` is pinned. The 4096/1024 result is
`54.47 output tok/s`, a strong mini-side improvement but still below the old
vLLM serving baseline `114.07 output tok/s`. Nsight now places sparse attention
and indexer/cache above Marlin WNA16, so the next target should move to
attention/indexer/cache or metadata/runtime overhead, not TARGET 07.4 precision
lanes.

## Primary References

Local mini-sglang:

- Main benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- Text correctness smoke: `benchmark/offline/deepseek_v4_text_smoke.py`
- DSV4 model: `python/minisgl/models/deepseek_v4.py`
- DSV4 attention: `python/minisgl/attention/deepseek_v4.py`
- DSV4 wrappers/kernels: `python/minisgl/kernel/deepseek_v4.py`
- Graph runner: `python/minisgl/engine/graph.py`
- DSV4 graph gate: `python/minisgl/engine/engine.py`
- Communication abstraction: `python/minisgl/distributed/impl.py`
- PyNCCL wrapper: `python/minisgl/kernel/pynccl.py`,
  `python/minisgl/kernel/csrc/src/pynccl.cu`
- sm80 kernel R&D record:
  `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md`
- TP8 benchmark baseline record:
  `prompts/TARGET_06_benchmark_sm80_baseline.md`

Old vLLM framework:

- Source root: `/workspace/vllm-dsv4-docker`
- Virtualenv: `/workspace/venvs/vllm-dsv4`
- Current vLLM helper scripts:
  `performance_milestones/vllm/scripts/`
- DSV4 model: `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- DSV4 attention:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- Fused MoE:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- CUDA graph / compile dispatcher:
  `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- Custom all-reduce:
  `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`
- Parallel communication custom ops:
  `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`

Known vLLM caveat: the sm80 reference sparse prefill path in
`deepseek_v4_attention.py` can materialize very large temporary tensors and has
already triggered OOM. Do not port that path as the mini default. Study vLLM's
scheduler, graph, communication, and MoE execution design first.

## Subtarget Index

Use separate Codex threads for these large milestones:

| Stage | Prompt | Purpose |
| --- | --- | --- |
| TARGET 07.1 | `prompts/TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md` | Fair mini/vLLM retest, nsys summaries, and execution-path diff. |
| TARGET 07.2 | `prompts/TARGET_07.2_dsv4_sm80_comm_cuda_graph.md` | Communication labeling, PyNCCL/custom all-reduce evaluation, and DSV4 decode CUDA graph enablement. |
| TARGET 07.25 | `prompts/TARGET_07.25_dsv4_sm80_vllm_subgraph_parity.md` | DeepSeek V4 sm80-only mini/vLLM subgraph parity map and paired microbench, used to rank the real remaining bottlenecks before more fixes. |
| TARGET 07.3 | `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md` | Exact MoE V2 plan/fusion after 07.25 identifies MoE as the top remaining bottleneck. |
| TARGET 07.35 | `prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md` | Re-run mini/vLLM parity after MoE V2, update the bottleneck ranking, and write the next focused performance plan before doing small optimizations. |
| TARGET 07.36 | `prompts/TARGET_07.36_dsv4_sm80_vllm_fused_moe_runner_adapt.md` | Adapt vLLM's standard FusedMoE runner shape into mini as the next exact-path baseline, then measure whether MoE runner structure still deserves more work. |
| TARGET 07.37 | `prompts/TARGET_07.37_dsv4_sm80_moe_backend_identification.md` | Identify the actual vLLM sm80 MoE expert backend and decide exact backend adaptation versus precision lane versus another target. |
| TARGET 07.38 | `prompts/TARGET_07.38_dsv4_sm80_moe_exact_backend_adapt.md` | Conditional implementation target for one vLLM-identified exact W4A16 MoE expert backend, only if TARGET 07.37 selects it. |
| TARGET 07.39 | `prompts/TARGET_07.39_dsv4_sm80_marlin_custom_op_bridge.md` | Completed bridge feasibility for locally installed vLLM Marlin custom ops; result is positive and recommends a mini-owned narrow csrc port target. |
| TARGET 07.391 | `prompts/TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port.md` | Completed mini-owned Marlin WNA16 csrc port and opt-in backend; macro improves to `54.47 output tok/s` but next bottleneck shifts to attention/indexer/cache. |
| TARGET 07.4 | `prompts/TARGET_07.4_dsv4_sm80_precision_lanes.md` | Precision-lane experiments: fp8/fp4 activation quantization and INT8 Tensor Core opt-in after bf16-direct is strong. |

Smaller work such as sqlite reporting helpers, benchmark flags, and README
updates may live inside the relevant subtarget rather than getting their own
thread.

## Current Sequencing After TARGET 07.36

TARGET 07.1, TARGET 07.2, TARGET 07.25, TARGET 07.3, TARGET 07.35, and
TARGET 07.36 now have recorded milestone artifacts. Do not continue expanding
those threads unless a baseline artifact is missing or a workload/config
mismatch is discovered.

After TARGET 07.391, carry these reference lines into the next target:

- mini-owned Marlin WNA16 exact backend: `54.47 output tok/s` and
  `61.41 decode tok/s` on 4096/1024/batch4 with TP8, page size 256,
  `--num-pages 128`, and CUDA graph replay;
- first hard victory line remains old serving baseline `114.07 output tok/s`;
- current risk order: sparse attention/indexer/cache first, then metadata and
  runtime overhead around graph replay and route handling, then communication,
  then any remaining expert backend polish;
- precision lanes are still not justified by the evidence because the Marlin
  exact backend is now fast enough that attention/indexer/cache dominate the
  short profile.

## Thread Stop Rules

Each subtarget thread must stop when it has achieved its evidence objective,
selected the next target, or shown that its scoped bottleneck is no longer the
best use of time. Do not keep optimizing a subtarget simply because another
small local improvement is possible.

Hard stop conditions for any implementation subtarget:

- the official 4096/1024/batch4 output throughput exceeds 114.07 tok/s and
  TP8 page-size-256 text smoke passes;
- the target's named bottleneck is no longer in the top two contributors after
  a new profile or parity report;
- two consecutive implementation cuts produce less than 5% macro throughput
  gain and less than 10% improvement in the targeted subgraph;
- the next proposed change is outside the target scope and lacks evidence for
  at least 5% expected E2E gain;
- correctness is unstable after one focused fix attempt, in which case record
  the blocker and hand off rather than layering more performance changes.

Soft stop guidance:

- small cleanups are allowed only when they remove a blocker, improve
  observability, or are on a measured critical path;
- after one large bottleneck is substantially reduced, rerun parity instead of
  continuing to polish that same area;
- every subtarget README should end with a clear `next target` decision and a
  short `do not continue here unless...` note.

## vLLM Comparison Policy

- Before implementing any large optimization, inspect the corresponding vLLM
  path and record whether mini should port, adapt, or intentionally diverge.
- Prefer borrowing proven design from vLLM when it applies to sm80 and mini's
  architecture. Avoid rebuilding a worse local version just because the code is
  nearby.
- After TARGET 07.2, do not continue optimizing isolated mini details unless a
  mini-vs-vLLM subgraph comparison shows that detail is on the critical path.
- For each major subgraph, compare operator boundary, tensor shape, precision
  lane, graph-capture behavior, kernel/operator names, communication count,
  CUDA stream usage, overlap behavior, and measured wall latency before deciding
  what to implement.
- Do not add a runtime dependency on vLLM. If code is ported, keep the copied
  surface narrow, preserve Apache-2.0 attribution, and adapt it to mini's local
  abstractions.
- Treat vLLM profile data as a guide, not an oracle. Some vLLM worker kernels
  may be hidden behind CUDA graph/multiprocess boundaries, so compare macro
  metrics, event counts, and code structure together.
- Treat multi-stream execution as a first-class possible gap. If vLLM overlaps
  metadata staging, communication, attention, MoE, or logits/sampling work
  across streams, record the event dependencies and estimate the wall-time
  benefit before calling the gap a pure kernel issue.

## Master Optimization Plan

1. Fair retest and observability.
   - Align mini/vLLM workloads, warmup, page/block size, TP size, output
     lengths, and chunked-prefill policy.
   - Produce one gap report with macro metrics, kernel counts, runtime counts,
     top kernels, communication count/bytes, and CUDA graph evidence.

2. Communication and CUDA graph.
   - Label all mini all-reduce/all-gather call sites by semantic source:
     embedding, attention output, MoE routed/shared, HC, lm_head.
   - Fix PyNCCL DSV4 correctness/coverage, especially fp32 lm_head logits.
   - Re-enable DSV4 decode CUDA graph for stable batch sizes `[1,2,4]`; keep
     prefill eager until metadata is stable.
   - Compare mini PyNCCL/symmetric-memory behavior with vLLM custom all-reduce.

3. vLLM subgraph parity and paired microbench.
   - Split DeepSeek V4 sm80 decode/prefill into matching mini/vLLM subgraphs:
     attention projection/norm/cache, sparse attention/indexer, MoE route and
     experts, shared experts, HC/RMSNorm, communication, logits/sampling, and
     graph metadata/replay.
   - Record CUDA stream topology and overlap opportunities for each subgraph,
     especially compute/communication overlap and metadata/sampling overlap.
   - For each subgraph, record whether vLLM should be `port`, `adapt`,
     `reject`, or `defer`.
   - Run paired microbench on the same shapes before doing more local
     implementation work.
   - Rank bottlenecks by contribution to the remaining 4096/1024 gap.

4. Small-kernel or subsystem reduction.
   - Only optimize small-kernel fragmentation after the subgraph map shows it is
     one of the top remaining bottlenecks.
   - Use selective `torch.compile`, Triton fusion, or vLLM-style boundaries only
     where the paired comparison predicts meaningful E2E impact.

5. MoE exact V2.
   - Introduce a mini-side MoE execution plan abstraction for route metadata,
     workspace, expert-token layout, and finalize/reduce boundaries.
   - Tighten grouped FP4 W13/W2 execution, reduce intermediate writes, reuse
     workspace, and use LUT/table-driven FP4/E8M0 decode where applicable.
   - Keep routed + shared expert outputs rank-local until the intended single
     TP reduce boundary.

6. Precision-lane experiments.
   - Keep bf16-direct as the first exact optimization lane.
   - Test fp8/fp4 activation quantization only after the bf16-direct path is
     strong enough to be a fair baseline.
   - Add INT8 Tensor Core MoE only behind an explicit opt-in toggle.
   - Compare quantized lanes against exact V2 for logits, top-k, text smoke,
     and E2E speed.
   - Do not promote quantized activation or INT8 lanes unless quality gates and
     4096/1024 performance gates are both satisfied.

7. Post-bottleneck re-parity and focused follow-up plan.
   - After MoE exact V2 or any other major bottleneck fix, rerun the fair macro
     workload and short profile.
   - Refresh the subgraph ranking before starting another implementation-heavy
     thread.
   - If the new top bottleneck is attention/cache/indexer, open a dedicated
     attention target rather than continuing MoE micro-tuning.
   - If no single bottleneck dominates, write a bounded small-optimization plan
     with explicit expected E2E contribution for each item.

## Global Acceptance Gates

- Gate A: fair 4096/1024 and 4096/128 reports exist for mini and vLLM with no
  hidden workload/config mismatch.
- Gate B: mini 4096/128 kernel count drops by at least 5x and all-reduce calls
  drop by at least 3x versus the recorded V1 MoE nsys trace.
- Gate C: exact mini 4096/1024/batch4 reaches at least 80 output tok/s.
- Gate D: before TARGET 07.3 implementation, a subgraph comparison identifies
  the top bottleneck group and explains at least the dominant remaining
  mini-vs-vLLM gap with paired measurements or explicit unknowns.
- Gate E: after each major bottleneck target, rerun or refresh parity before
  opening a thread for non-dominant small optimizations.
- Final Gate: exact mini, or an explicitly approved opt-in path, exceeds
  114.07 output tok/s on 4096/1024/batch4 and passes TP8 page-size-256 text
  smoke.

## Non-Goals

- Do not make vLLM a runtime dependency of mini-sglang.
- Do not depend on DeepGEMM or FlashMLA sm90/sm100-only cubins for the sm80
  default path.
- Do not port vLLM's sm80 reference sparse prefill implementation as a default
  if it retains the OOM-prone large materialization behavior.
- Do not promote approximate INT8/FP8/FP4 behavior based only on microbench
  speed.
- Do not treat vLLM's precision implementation as mandatory for mini's
  bf16-direct exact path.

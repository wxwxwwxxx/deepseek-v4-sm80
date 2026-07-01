# TARGET 07.36: DSV4 sm80 vLLM FusedMoE Runner Adaptation

## Goal

Adapt the old vLLM DeepSeek V4 sm80 `FusedMoE` runner design into mini-sglang
as a mini-owned exact-path baseline, then measure whether this structural
baseline closes the post-TARGET 07.35 MoE gap.

This target is intentionally not another round of local route/workspace micro
cleanup.  The purpose is to borrow the strongest applicable vLLM design first,
keep mini's default precision semantics explicit, and only optimize locally
after the vLLM-shaped baseline is measured.

Default precision policy for this target:

- keep mini's exact bf16-direct activation path;
- keep model-original fp32 math as fp32 unless an explicit TF32 experiment is
  opened elsewhere;
- do not add activation quantization, INT8 MoE, MXFP4/FP8 cache promotion, or a
  vLLM runtime dependency;
- defer vLLM precision-lane behavior to TARGET 07.4.

## Start Point

TARGET 07.35 is complete and recorded in
`performance_milestones/target07_post_moe_reparity/`.

Important facts to carry forward:

- current exact mini variant:
  `v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`;
- 4096/1024/batch4 exact V2: about `17.8009` E2E output tok/s and `19.9037`
  decode tok/s under the TARGET 07.35 smoke macro policy;
- 4096/128/batch4 exact V2: about `10.7751` E2E output tok/s and `19.8442`
  decode tok/s;
- TP8 page-size-256 text smoke passes, with 9/9 decode graph replays and no
  eager decode fallback;
- TARGET 07.3 MoE V2 route plan/workspace and bf16-output SwiGLU cuts were
  correct but did not improve decode;
- TARGET 07.25 Nsight evidence still shows mini grouped FP4 W13/W2 as the
  largest measured kernel-time category, with sparse attention/cache/indexer as
  the next major risk;
- old vLLM fair reference remains far ahead, but it uses a different precision
  policy, so precision behavior must be called out whenever it is adapted.

## Primary References

Mini:

- master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- post-MoE re-parity:
  `performance_milestones/target07_post_moe_reparity/README.md`
- vLLM runner integration sketch:
  `performance_milestones/target07_post_moe_reparity/summaries/vllm_fused_moe_runner_integration.md`
- subgraph parity:
  `performance_milestones/target07_subgraph_parity/README.md`
- MoE V2 results: `performance_milestones/target07_moe_v2/RESULTS.md`
- mini DSV4 model: `python/minisgl/models/deepseek_v4.py`
- mini DSV4 wrappers: `python/minisgl/kernel/deepseek_v4.py`
- mini DSV4 Triton kernels: `python/minisgl/kernel/triton/deepseek_v4.py`
- MoE microbench: `benchmark/offline/deepseek_v4_moe_route_microbench.py`
- macro benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- text smoke: `benchmark/offline/deepseek_v4_text_smoke.py`

vLLM:

- source root: `/workspace/vllm-dsv4-docker`
- DSV4 model:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- standard FusedMoE layer:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/layer.py`
- MoE runner:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- shared experts scheduler:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/shared_experts.py`
- modular prepare/finalize and experts interfaces:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- no-DP/EP prepare/finalize:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py`
- classic Triton fused MoE and expert assignment:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/fused_moe.py`
- MXFP4 backend selector, for reference only in this exact target:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`

Known vLLM path decision:

- adapt standard `FusedMoE` runner shape;
- reject `DeepseekV4MegaMoEExperts` for sm80;
- defer MXFP4/FP8 precision semantics to TARGET 07.4 unless this target produces
  evidence that exact runner adaptation is exhausted.

## Adaptation Plan

1. Build a concrete mini-vs-vLLM MoE component map before editing kernels.
   - Map vLLM `DeepseekV4MoE -> FusedMoE -> MoERunner -> prepare -> experts ->
     finalize -> shared experts -> reduce` to mini's `DSV4MoE`,
     `DSV4FusedRoutedExperts`, `DSV4SharedExperts`, and
     `DSV4MoEExecutionPlan`.
   - Record which vLLM behavior is ported, adapted, rejected, or deferred.
   - Preserve Apache-2.0 attribution if any vLLM code is copied instead of only
     used as a design reference.

2. Add a mini-owned `DSV4FusedMoERunner`.
   - Put route selection, route metadata ownership, workspace sizing, expert
     application, top-k weighting/finalize, shared expert scheduling, and final
     reduce decision behind one runner boundary.
   - Keep the current `DSV4MoEGate` for fp32 route weights and int expert ids.
   - Wrap the existing exact grouped FP4 W13/SwiGLU/W2 kernels first; do not
     change expert precision in the first cut.
   - Keep V1/V2 paths available as fallbacks.
   - Gate the runner with an explicit opt-in toggle, for example
     `MINISGL_DSV4_SM80_MOE_VLLM_RUNNER=1`, and add a benchmark variant for it.

3. Match vLLM's standard no-DP/EP modular boundary.
   - `prepare`: create or reuse route metadata and expert-token layout.
   - `experts`: call exact mini grouped FP4 expert compute through a modular
     interface that exposes workspace shapes and dtypes.
   - `finalize`: apply route weights and reduce top-k routed output to `[M, H]`.
   - `shared experts`: keep serial in the first cut, but place the scheduling
     decision inside the runner so aux-stream overlap can be added later without
     another model-level rewrite.
   - `reduce`: keep one late TP all-reduce after routed plus shared local sum
     unless the runner proves the routed output is already reduced.

4. Prefer vLLM's proven fast-path ideas over new local inventions.
   - Evaluate vLLM's small decode expert-assignment fast path that avoids full
     sorted-token metadata when route count is tiny.
   - Keep this fast path as a measured second cut only if route metadata appears
     in the runner profile or microbench.
   - Do not spend a thread on it if W13/W2 expert compute still dominates.

5. Measure immediately after the first runner cut.
   - Correctness: unit tests for routed-only, shared-only, routed+shared, hash
     routing, and correction-bias routing; TP8 page-size-256 text smoke.
   - Microbench: extend `deepseek_v4_moe_route_microbench.py` with
     `runner_prepare_ms`, `runner_experts_ms`, `runner_finalize_ms`,
     `runner_shared_ms`, and `runner_total_ms`.
   - Macro: run 4096/128/batch4 profile-equivalent and 4096/1024/batch4 official
     workloads.
   - Profile: capture short 4096/128 Nsight after the first runner cut, or record
     why it was not possible.

6. Decide the second cut from data only.
   - If the runner cut improves 4096/1024 macro by at least 5 percent or routed
     MoE subgraph by at least 10 percent, allow one focused follow-up cut:
     shared-expert aux-stream overlap, vLLM-style expert assignment fast path, or
     exact grouped FP4 expert-kernel replacement, whichever the profile supports.
   - If the runner cut is below those thresholds, stop exact runner work.  Do not
     keep polishing wrapper boundaries.
   - If W13/W2 remain dominant after runner adaptation, open a backend/precision
     decision: either an exact expert-kernel replacement target or TARGET 07.4
     for vLLM MXFP4/INT8/FP8 opt-in lanes.
   - If sparse attention/cache/indexer overtakes MoE, open a dedicated
     attention/cache/indexer target instead.

## Stop Conditions

Stop this target when any condition is met:

- exact or explicitly approved opt-in mini exceeds `114.07` output tok/s on
  4096/1024/batch4 and TP8 page-size-256 text smoke passes;
- the first vLLM-shaped runner cut produces less than 5 percent macro gain and
  less than 10 percent routed-MoE subgraph gain;
- fresh 4096/128 profile shows attention/cache/indexer clearly ahead of MoE;
- the next proposed change requires changing default precision semantics;
- correctness becomes unstable after one focused fix attempt;
- no vLLM design element remains to adapt without becoming a local kernel rewrite
  or a precision-lane experiment.

## Done Criteria

- `performance_milestones/target07_vllm_fused_moe_runner/README.md` records the
  component map, implementation cuts, correctness, macro results, and next
  decision.
- A mini-owned runner variant exists behind an explicit toggle or the target
  records why implementation was rejected before coding.
- TP8 text smoke passes for any implemented runner variant.
- MoE microbench includes runner prepare/experts/finalize/shared breakdown.
- 4096/128 and 4096/1024 benchmark artifacts are recorded.
- The final decision says one of:
  - continue with one measured runner follow-up cut;
  - open attention/cache/indexer target;
  - open exact expert-kernel backend target;
  - open TARGET 07.4 precision lane;
  - stop because the win line has been reached.

## Non-Goals

- Do not add vLLM as a runtime dependency.
- Do not port `DeepseekV4MegaMoEExperts` for sm80.
- Do not promote vLLM MXFP4/FP8 precision semantics into mini's exact default.
- Do not implement INT8 Tensor Core MoE in this target.
- Do not optimize sparse attention, cache layout, custom all-reduce, or sampling
  here unless the change is strictly required to measure the runner.
- Do not continue route/workspace-only MoE cleanup after the first runner cut
  misses the stop thresholds.

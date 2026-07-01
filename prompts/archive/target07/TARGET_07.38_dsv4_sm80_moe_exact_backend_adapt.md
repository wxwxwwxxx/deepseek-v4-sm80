# TARGET 07.38: DSV4 sm80 MoE Exact Expert Backend Adaptation

## Goal

Adapt one vLLM-identified exact MoE expert backend into mini-sglang as an
opt-in backend for the existing `DSV4FusedMoERunner`, then measure whether W13/W2
kernel time and 4096/1024 throughput improve enough to continue exact MoE work.

Start this target only if TARGET 07.37 classifies a vLLM backend as an
`exact_candidate`.

## Required Start Conditions

All must be true before implementation:

- TARGET 07.37 has a README and final decision selecting this target;
- the selected backend is W4A16 or equivalent with bf16 activations and no
  required activation quantization;
- expected routed-MoE microbench improvement is at least `1.5x`;
- required dependencies and source files are present locally or can be replaced
  with a narrow mini-owned implementation;
- default mini precision policy remains unchanged.

If any condition fails, do not start this target. Use TARGET 07.4 for precision
lanes or open a local exact-kernel plan instead.

## Primary References

- Backend identification:
  `performance_milestones/target07_moe_backend_identification/README.md`
- Runner milestone:
  `performance_milestones/target07_vllm_fused_moe_runner/README.md`
- mini runner/model: `python/minisgl/models/deepseek_v4.py`
- mini wrapper API: `python/minisgl/kernel/deepseek_v4.py`
- mini Triton MoE kernels: `python/minisgl/kernel/triton/deepseek_v4.py`
- MoE microbench: `benchmark/offline/deepseek_v4_moe_route_microbench.py`
- macro benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- vLLM backend files selected by TARGET 07.37

## Plan

1. Create `performance_milestones/target07_moe_exact_backend_adapt/`.
   - Record the selected backend, source references, copied/adapted files, and
     Apache-2.0 attribution if any vLLM code is copied.
   - Save microbench, macro, text-smoke, and short Nsight artifacts.

2. Add a narrow opt-in backend surface.
   - Use an explicit env or variant, for example
     `MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=<backend_name>`.
   - Default remains the current grouped FP4 backend.
   - 07.36 runner remains the integration boundary; do not rewrite route,
     prepare, shared experts, or final reduce unless the backend requires a
     minimal adapter.
   - Fallback to current grouped FP4 backend if the candidate backend is
     unsupported for a shape.

3. Implement only the selected exact backend.
   - Port or reimplement the required weight transform, scale layout, workspace
     allocation, route metadata adapter, and W13/W2 invocation.
   - Keep activations bf16.
   - Keep top-k weighting and route sum semantics identical to the current exact
     path within existing tolerances.
   - Do not add activation quantization, INT8, MXFP8, or FP8 cache behavior.

4. Extend benchmark variants and observability.
   - Add a perf-matrix variant for the backend plus graph/runner toggles.
   - Extend MoE microbench with backend-specific W13, activation, W2, route_sum,
     and total timings.
   - Label fallback/unsupported backend skips so a macro pass cannot silently run
     the old backend.

5. Measure and decide.
   - Run unit/wrapper tests, TP8 text smoke, DSV4-like MoE microbench,
     4096/128/batch4 profile-equivalent, 4096/1024/batch4 macro, and short
     4096/128 Nsight.
   - Compare against the 07.36 runner baseline.
   - Continue exact backend work only if the first cut reaches at least `10%`
     macro gain, at least `20%` routed-MoE microbench gain, and at least `1.5x`
     W13/W2 summed kernel-time reduction.

## Stop Conditions

Stop immediately if:

- correctness or text smoke fails after one focused fix attempt;
- the backend silently falls back on official macro workloads;
- 4096/1024 macro gain is below `10%` and routed-MoE microbench gain is below
  `20%`;
- W13/W2 summed kernel time does not drop by at least `1.5x`;
- implementation requires activation quantization or another precision-lane
  semantic change;
- attention/cache/indexer becomes the clear top bottleneck in the new profile.

## Done Criteria

- Opt-in backend variant is implemented or rejected with a precise blocker.
- Correctness and TP8 text smoke are recorded.
- MoE microbench and macro results are recorded against the 07.36 baseline.
- Short Nsight summary records W13/W2, NCCL, sparse attention, and indexer time.
- README final decision says one of:
  - continue one exact backend follow-up cut;
  - stop exact backend and move to TARGET 07.4;
  - stop MoE and open attention/cache/indexer target;
  - victory line reached and stabilization is next.

## Non-Goals

- Do not investigate backend selection here; TARGET 07.37 must already have made
  that decision.
- Do not add multiple competing backends.
- Do not change default precision.
- Do not add vLLM as a runtime dependency.
- Do not resume MoE runner/wrapper cleanup.

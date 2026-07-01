# TARGET 07.35: DSV4 sm80 Post-MoE Re-Parity

## Goal

Re-evaluate mini-sglang versus the old vLLM DeepSeek V4 Flash path after
TARGET 07.3 MoE exact V2 lands one serious implementation cut.

This target is complete when the new bottleneck ranking is recorded and the
next focused implementation plan is chosen. It exists to prevent the project
from spending another long thread on non-dominant micro-optimizations after the
main MoE bottleneck has moved.

## When To Start

Start this target after TARGET 07.3 has one of the following outcomes:

- exact 4096/1024/batch4 output throughput improves by at least 1.3x over the
  post-07.2 best exact baseline of about 25.3 output tok/s;
- MoE routed W13/W2 summed kernel time drops by at least 2x in the short
  4096/128 profile;
- TARGET 07.3 hits a stop condition and needs a fresh ranking before more work.

Do not start this target before a real MoE V2 attempt exists. If TARGET 07.3
only changes observability or benchmark scripts, finish TARGET 07.3 first.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- MoE V2 target: `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`
- previous subgraph parity:
  `performance_milestones/target07_subgraph_parity/README.md`
- TARGET 07.2 comm/graph record:
  `performance_milestones/target07_comm_graph/README.md`
- mini benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- text smoke: `benchmark/offline/deepseek_v4_text_smoke.py`
- vLLM source root: `/workspace/vllm-dsv4-docker`

## Plan

1. Create `performance_milestones/target07_post_moe_reparity/`.
   - Keep large raw profiles as symlinks under `raw/`.
   - Store small JSON/markdown summaries under `summaries/`.
   - Add a README with baseline, new results, bottleneck ranking, and next
     target decision.

2. Freeze the new mini baseline.
   - Record exact variant name, toggles, git status, model path, page size,
     TP size, and correctness smoke artifact.
   - Run the official 4096/1024/batch4 macro workload.
   - Run the 4096/128/batch4 short profile or reuse a fresh nsys artifact from
     TARGET 07.3 if it has the same workload shape.

3. Re-rank the gap.
   - Compare against the previous TARGET 07.25 ranking:
     1. MoE routed experts and MoE execution boundary;
     2. sparse attention/indexer/cache layout;
     3. scheduling/graph/multi-stream overlap;
     4. communication/reduce boundary;
     5. precision lane;
     6. HC/RMSNorm/final/sampling.
   - Record whether MoE is still top, moved below attention/cache, or exposed
     communication/overlap as the next limiting factor.
   - Use new measurements first; keep the old vLLM parity results only as a
     reference when vLLM rerun is not necessary.

4. Decide the next focused plan.
   - If exact mini exceeds 114.07 output tok/s and text smoke passes, freeze the
     win and write a cleanup/stabilization plan instead of chasing more speed in
     the same thread.
   - If MoE remains top and another clearly scoped MoE cut has at least 10%
     expected E2E impact, extend TARGET 07.3 with that single cut.
   - If attention/cache/indexer is top, create a dedicated attention/cache
     target before implementation.
   - If communication or overlap becomes top, create a communication/stream
     target with measured call counts, bytes, and overlap opportunities.
   - If the remaining gap is dominated by vLLM precision lanes, move to
     TARGET 07.4.
   - If no single item dominates, write a bounded small-optimization plan with
     per-item expected E2E gain and a stop rule.

## Stop Conditions

Stop this target as soon as the new ranking and next target decision are
recorded. Do not implement kernels here.

Do not spend time on small cleanups unless they are required to make the
re-parity measurement possible. If the new report cannot explain the gap, record
the unknowns and the exact profiler or instrumentation needed next.

## Done Criteria

- `performance_milestones/target07_post_moe_reparity/README.md` exists.
- New exact mini 4096/1024/batch4 macro result is recorded.
- New 4096/128 short profile or equivalent fresh profile is recorded.
- The post-MoE bottleneck ranking is updated.
- The next target is selected with one of:
  - continue TARGET 07.3 with one scoped MoE cut;
  - open an attention/cache/indexer target;
  - open a communication/stream-overlap target;
  - start TARGET 07.4 precision lanes;
  - write a bounded small-optimization plan.

## Non-Goals

- Do not implement MoE, attention, communication, or precision kernels here.
- Do not rerun vLLM unless mini's workload/config changed in a way that makes
  the previous fair reference invalid.
- Do not continue polishing MoE after it is no longer a top bottleneck.

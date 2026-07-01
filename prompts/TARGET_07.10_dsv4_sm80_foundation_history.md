# TARGET 07.10: DSV4 SM80 Foundation And Parity History

## Status

Completed history merge for the early TARGET 07 work.

This file replaces the need to read the full original prompt chain for the
foundation phase:

- `TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md`
- `TARGET_07.2_dsv4_sm80_comm_cuda_graph.md`
- `TARGET_07.25_dsv4_sm80_vllm_subgraph_parity.md`

The original prompt files now live under `prompts/archive/target07/` as
archival references for exact commands and thread-local details.

## Motivation

The first TARGET 07 problem was not "which kernel should be optimized?", but
"what is the fair gap?".  mini-sglang and the old vLLM-based framework differed
in workload setup, graph capture, process model, cache layout, precision lane,
and profiling visibility.  Optimizing small local kernels before aligning these
facts led to low-value work.

The foundation phase therefore had three goals:

1. build a fair mini/vLLM benchmark and profile comparison;
2. enable enough graph/communication observability to stop counting noise as
   bottleneck;
3. split the DeepSeek V4 sm80 decode path into comparable subgraphs and rank
   the real remaining gaps.

## Completed Milestones

### TARGET 07.1: Fair Rebench And vLLM Diff

Artifacts:

- `performance_milestones/target07_vllm_gap/`
- `performance_milestones/vllm/`

Main conclusion:

- The old vLLM-based path was far ahead on the same 4096/128 and
  4096/1024 batch4 workload shapes.
- mini had much higher kernel/runtime event counts and was still dominated by
  fragmented fallback paths, communication, and MoE expert execution.
- The vLLM source root is `/workspace/vllm-dsv4-docker`; its virtualenv is
  `/workspace/venvs/vllm-dsv4`.

### TARGET 07.2: Communication And CUDA Graph

Artifacts:

- `performance_milestones/target07_comm_graph/`

Main conclusion:

- DSV4 communication sites were labeled by semantic source.
- PyNCCL coverage and correctness improved.
- DSV4 decode CUDA graph replay was enabled for stable batch sizes.
- The best exact 4096/1024/batch4 line moved to about `25.3 output tok/s`.
- After the big graph/communication fixes, further graph-surface cleanup gave
  tiny gains.  This phase should not be expanded unless a new profile shows
  communication or graph replay has returned as a top-two bottleneck.

### TARGET 07.25: Subgraph Parity

Artifacts:

- `performance_milestones/target07_subgraph_parity/`

Main conclusion:

The remaining gap was ranked as:

1. MoE routed experts and MoE execution boundary;
2. sparse attention/indexer/cache layout;
3. scheduling, graph, and multi-stream overlap;
4. communication/reduce boundary;
5. vLLM-only precision lane;
6. HC/RMSNorm/final/sampling.

This ranking selected the MoE chain as the next major implementation phase.

## Persistent Rules From This Phase

- Do fair macro first, then subgraph attribution, then implementation.
- Prefer vLLM-informed boundaries when they apply to sm80 and mini's runtime.
- Do not optimize isolated mini details unless a parity report places them on
  the critical path.
- Treat vLLM profile data as a guide, not an oracle; some vLLM child-process
  CUDA graph work is hard to attribute from old SQLite profiles.
- Each implementation target should stop after it proves or disproves its named
  bottleneck.

## Do Not Continue Here Unless

- a workload/config mismatch is found in the old fair comparison;
- a new profiler setup makes the old vLLM trace substantially more complete;
- a later change invalidates the current TP8/page-size-256 benchmark harness.

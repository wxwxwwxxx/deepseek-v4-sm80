# TARGET 07.25: DSV4 sm80 vLLM Subgraph Parity and Microbench

## Goal

Build a complete DeepSeek V4 Flash sm80-only comparison between mini-sglang and
the old vLLM-based implementation before doing more large implementation work.

This target is complete when the remaining mini-vs-vLLM performance gap is
allocated to named subgraphs with paired measurements, and the next
implementation target is chosen by measured severity rather than local
guesswork.

## Why This Target Exists

TARGET 07.2 produced meaningful infrastructure wins: PyNCCL correctness, DSV4
decode CUDA graph replay, graph-node attribution, HC/RMSNorm helpers, and
vLLM-aligned attention-boundary experiments. The best exact path reached about
25.3 output tok/s on 4096/1024/batch4, up from the fair mini V1 baseline of
about 10.6 output tok/s.

However, the old vLLM path remains much faster. TARGET 07.2 also showed that
many late-stage local changes are only tiny positives or even negative despite
positive isolated microbench results. The next step is therefore not another
small mini-only optimization. It is a structured mini-vs-vLLM subgraph map.

## Completion Update

This target now has a completed milestone report at
`performance_milestones/target07_subgraph_parity/README.md`.

The recorded bottleneck ranking for 4096/1024/batch4 is:

1. MoE routed experts and MoE execution boundary;
2. sparse attention/indexer/cache layout;
3. scheduling/graph/multi-stream overlap;
4. communication/reduce boundary;
5. vLLM-only precision lane;
6. HC/RMSNorm/final/sampling.

The selected next target is TARGET 07.3 MoE exact V2. Do not keep expanding this
target with additional microbenchmarks unless a later implementation invalidates
the current ranking. After MoE V2, use
`prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md` for the next parity pass.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- TARGET 07.1 fair rebench:
  `prompts/TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md`
- TARGET 07.2 comm/graph record:
  `performance_milestones/target07_comm_graph/README.md`
- vLLM source root: `/workspace/vllm-dsv4-docker`
- vLLM virtualenv: `/workspace/venvs/vllm-dsv4`
- mini benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- mini DSV4 model: `python/minisgl/models/deepseek_v4.py`
- mini DSV4 attention: `python/minisgl/attention/deepseek_v4.py`
- mini DSV4 kernels: `python/minisgl/kernel/deepseek_v4.py`

vLLM code paths to inspect first:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`

## Scope

Only compare DeepSeek V4 Flash on single-node TP8 A100/sm80.

Do not generalize this target to other models, other architectures, online
serving policy, expert parallelism, or sm90/sm100 kernels. vLLM is a reference
implementation and source of proven design ideas, not a runtime dependency.

## Subgraph Taxonomy

Create a table with the following columns for every subgraph:

- subgraph name;
- mini entrypoint and operator/kernel list;
- vLLM entrypoint and operator/kernel list;
- tensor shapes for 4096/128 and 4096/1024 decode;
- precision lane;
- communication operations and counts;
- CUDA graph behavior;
- stream usage and overlap behavior;
- current mini latency;
- current vLLM latency;
- measured or estimated overlap gain;
- ratio;
- decision: `port`, `adapt`, `reject`, or `defer`;
- evidence artifact path.

Required subgraphs:

1. Scheduler and graph surface.
   - decode batch preparation;
   - page/KV metadata staging;
   - graph input copy/replay;
   - logits/sampling surface.
   - stream creation, stream selection, event synchronization, and cross-stream
     overlap policy.

2. Attention front projection and cache insert.
   - `wq_a + wkv` projection;
   - q/KV RMSNorm;
   - q RoPE;
   - KV RoPE/cache store;
   - `wq_b`, `wo_a`, `wo_b`;
   - fused or shared activation/weight preparation.

3. Sparse attention and indexer.
   - C4/C128/SWA attention;
   - top-k/indexer query path;
   - sparse metadata construction;
   - prefill reference path caveats.

4. MoE route and routed experts.
   - gate and top-k;
   - route metadata;
   - W13;
   - activation/finalize;
   - W2;
   - route sum;
   - TP reduce boundary.

5. Shared experts.
   - gate/up projection;
   - activation;
   - down projection;
   - interaction with routed expert output and reduce boundary.

6. HC/RMSNorm/final layers.
   - HC pre/post/head;
   - RMSNorm;
   - embedding and lm_head.

7. Communication.
   - all-reduce/all-gather call sites;
   - tensor sizes and dtypes;
   - communicator implementation;
   - whether communication is inside graph replay;
   - whether communication is overlapped with compute on another stream.

8. Multi-stream overlap.
   - vLLM stream ownership and lifetime for DeepSeek V4 decode/prefill;
   - CUDA events and synchronization edges between streams;
   - overlap between metadata staging, communication, attention, MoE, and
     sampling/logits work;
   - whether mini currently serializes work that vLLM overlaps;
   - estimated benefit from overlap, reported as wall-time reduction rather
     than only summed kernel-time reduction.

## Paired Microbench Policy

- Use the same subgraph boundary for mini and vLLM. If exact boundary matching
  is impossible, record the mismatch and benchmark the nearest comparable
  boundary.
- Benchmark decode-like `tokens=4` and prefill/chunk-like `tokens=4096` where
  the subgraph exists in both frameworks.
- Prefer shape-equivalent deterministic synthetic tensors for first-pass speed
  comparison.
- Use real checkpoint weights only for a small number of top candidate
  subgraphs where weight layout or quantization packing is the suspected
  difference.
- Keep mini and vLLM microbench scripts separate if the Python environments are
  incompatible. Store commands and small JSON summaries under
  `performance_milestones/target07_subgraph_parity/`.
- Include warmup, CUDA synchronization, repeat count, dtype, shape, and active
  toggles in every microbench JSON.
- If a vLLM subgraph uses multiple CUDA streams, record stream count, stream
  roles, event dependencies, and whether the paired mini benchmark is serial or
  overlapped. Use CUDA events or nsys timelines to report both summed kernel
  time and wall time, because overlap can make summed kernel time misleading.

## Plan

1. Create `performance_milestones/target07_subgraph_parity/`.
   - Keep large raw profiles as symlinks under `raw/`.
   - Store small JSON/markdown summaries under `summaries/`.
   - Add a README with the subgraph map and ranked bottleneck list.

2. Freeze the comparison baselines.
   - mini baseline: best exact post-07.2 graph variant from
     `performance_milestones/target07_comm_graph`.
   - vLLM baseline: fair 4096/128 and 4096/1024 runs from the vLLM comparison
     scripts.
   - Record if the vLLM run uses precision behavior that mini intentionally
     does not match in bf16-direct mode.

3. Build the subgraph map.
   - Read both implementations and fill the taxonomy table before writing new
     kernels.
   - Use existing 07.2 node-trace summaries for mini graph-body attribution.
   - Inspect vLLM stream usage in the scheduler, graph runner, communication
     path, attention path, and MoE path. Mark any mini-vs-vLLM difference as an
     overlap opportunity before treating it as a pure kernel gap.
   - Add vLLM NVTX/profiler instrumentation only if code inspection plus
     existing nsys data cannot identify a boundary.

4. Build paired microbench scripts.
   - Start with MoE route/routed experts, attention front projection/cache
     insert, sparse attention/indexer, shared experts, and communication.
   - For each subgraph, produce mini and vLLM timing JSON with the same shape
     labels.
   - Where vLLM uses multiple streams, add an overlap-aware benchmark or nsys
     slice that reports critical-path wall time, serialized wall time when
     feasible, and the estimated overlap gain.
   - Include a simple roofline-style classification: compute-bound,
     memory-bandwidth-bound, launch/graph-bound, communication-bound, or
     unknown.

5. Rank bottlenecks.
   - Estimate each subgraph's contribution to the remaining 4096/1024 gap.
   - Separate structural gaps from microkernel gaps:
     - structural: vLLM uses a different boundary/fusion/graph plan;
     - microkernel: same boundary, vLLM kernel is faster;
     - scheduling: same compute but different batching/chunking/graph replay;
     - overlap: vLLM hides work with multiple streams while mini serializes it;
     - communication: different collective count, size, implementation, or
       compute/communication overlap;
     - precision: different quantization or dtype lane.
   - Choose the next implementation target only after this ranking exists.

## Decision Rules

- `port`: vLLM code/design is directly applicable to mini on sm80 and does not
  violate the precision policy.
- `adapt`: vLLM design is right, but the code must be rewritten for mini's
  wrappers, cache layout, or precision policy.
- `reject`: vLLM path is sm90/sm100-only, OOM-prone on sm80, precision-mismatched
  for the current lane, or incompatible with mini without large unrelated
  dependencies.
- `defer`: evidence is insufficient or the subgraph is not a top contributor.

## Done Criteria

- `performance_milestones/target07_subgraph_parity/README.md` contains a
  completed subgraph map for DeepSeek V4 sm80.
- Each required subgraph has either paired microbench data or a documented
  reason why paired measurement is not currently feasible.
- The top bottleneck group is identified with supporting measurements.
- The next target is explicitly selected:
  - TARGET 07.3 MoE exact V2;
  - a new attention/cache-insert target;
  - a communication/reduce-boundary target;
  - or a precision-lane target.
- TARGET 07 master doc is updated with the ranking and chosen next step.

## Stop Conditions

Stop this target once the subgraph map, bottleneck ranking, and next target
decision exist. Do not add kernels here.

Only reopen the parity work when:

- TARGET 07.3 or another major implementation target materially changes the
  execution path;
- a profile artifact is found to be invalid or unfair;
- the next target cannot be chosen because the current ranking lacks evidence.

## Non-Goals

- Do not implement MoE V2, new attention kernels, or new precision lanes in
  this target.
- Do not add vLLM as a mini runtime dependency.
- Do not attempt to make mini match vLLM precision behavior in the bf16-direct
  exact lane.
- Do not port vLLM's sm80 sparse prefill reference path if it retains the
  OOM-prone large materialization behavior.

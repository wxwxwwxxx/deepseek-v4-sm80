# TARGET 10.1: DSV4 SM80 Communication Path Parity With vLLM

## Status

Next executable TARGET 10 child target.

Run this before any PyNCCL, symmetric-memory, NCCL tuning, overlap, or
attention-kernel changes.

## Goal

Build a precise mini-vs-vLLM communication map for DeepSeek V4 Flash on TP8
SM80.

The goal is not to optimize communication yet.  The goal is to answer:

```text
Does mini communicate the same tensors, the same number of times, at the same
logical boundaries, with comparable bytes and graph placement as vLLM?
```

If the answer is no, prefer adapting the vLLM-aligned path before tuning the
communication backend.

## Required Inputs

Project and reference paths:

- mini: `/workspace/mini-sglang`
- vLLM reference: `/workspace/vllm-dsv4-docker`
- vLLM virtualenv: `/workspace/venvs/vllm-dsv4`
- vLLM Python: `/workspace/venvs/vllm-dsv4/bin/python`
- mini Python: system interpreter from `/workspace/mini-sglang`
- model: `/models/DeepSeek-V4-Flash`

Mini references:

- `python/minisgl/distributed/impl.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/layers/embedding.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `performance_milestones/target08_post_prefix_reprofile/README.md`

vLLM references:

- `performance_milestones/target07_vllm_gap/README.md`
- `performance_milestones/target07_vllm_gap/scripts/run_vllm_4096x1024_bs4.sh`
- `performance_milestones/target07_vllm_gap/scripts/nsys_vllm_4096x128_bs4_fair.sh`
- `performance_milestones/vllm/scripts/vllm_env.sh`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/logits_processor.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/communication_op.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`

## Baseline

Use the TARGET 08 prefix baseline unless a probe explicitly needs a smaller
case:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Minimum mini scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16` as a prefix-health check.

Use smaller probes only to classify per-layer or per-owner behavior, not as the
sole evidence for the final decision.

## Work Plan

### 1. Static Source Parity

Create a table that maps mini owner labels to vLLM implementation boundaries.

At minimum include:

- embedding all-reduce;
- attention `wo_b` row-parallel all-reduce;
- MoE routed/shared/reduce-once reductions;
- lm-head/logits all-gather or gather;
- any reduce-scatter, all-gather, or extra custom collective in vLLM that mini
  does not use.

For every row, record:

- mini file/function;
- vLLM file/function;
- op type;
- dtype;
- expected tensor shape as a function of batch/token count and hidden size;
- number of times per layer or per request;
- whether it runs inside decode graph replay.

### 2. Mini Runtime Communication Census

Use mini's `DistributedCommunicator` stats and owner timing to collect:

- per-label collective count;
- bytes;
- dtype;
- input/output shape;
- per-owner wall time when owner timing is enabled;
- graph replay/eager status.

If the existing report is missing any field, add the smallest instrumentation
needed to expose it.  Keep instrumentation opt-in or benchmark-local.

### 3. vLLM Runtime Census, With Static Fallback

Use runtime measurement as the first-class path.  vLLM is expected to be
runnable in this container through:

```bash
source /workspace/venvs/vllm-dsv4/bin/activate
source /workspace/mini-sglang/performance_milestones/vllm/scripts/vllm_env.sh
setup_vllm_runtime_env
cd /workspace/vllm-dsv4-docker
```

Mini should continue to use the system interpreter from
`/workspace/mini-sglang`.

Useful options:

- lightweight monkeypatch/wrapper around vLLM `get_tp_group().all_reduce`,
  `all_gather`, and `reduce_scatter`;
- nsys/NVTX using the TARGET 07 fair-run scripts as templates;
- source-derived shape/count table from `RowParallelLinear`, FusedMoE runner,
  embedding, and logits processor.

If a specific runtime probe or monkeypatch fails, first check the TARGET 07
vLLM run history and scripts listed above.  If the failure is still specific to
that probe, produce a source-derived expected census and mark exactly which
rows are static/inferred.  Do not silently replace runtime evidence with static
evidence.

### 4. Compare And Classify Differences

Produce a parity table with one of:

- `match`;
- `mini extra`;
- `vLLM extra`;
- `shape/bytes mismatch`;
- `dtype mismatch`;
- `backend-only difference`;
- `unknown`.

For each mismatch, decide whether it is:

- likely performance material;
- correctness/architecture required;
- an artifact of benchmark differences;
- a candidate for mini to adapt from vLLM.

### 5. Optional Minimal Adaptation

Only implement an adaptation if TARGET 10.1 finds a clear path-level mismatch
that is low risk and directly vLLM-aligned.

Examples:

- remove an extra collective;
- move a reduction to the same boundary as vLLM;
- change logits gather behavior if vLLM has a cheaper equivalent and correctness
  is preserved.

Do not tune NCCL/PyNCCL in this target.

## Deliverables

Write:

```text
performance_milestones/target10_comm_path_parity_vllm/README.md
```

Include:

- mini-vs-vLLM communication path table;
- per-label mini count/bytes/timing table;
- vLLM runtime count/bytes table, with any static fallback rows explicitly
  marked;
- graph replay/eager status;
- list of mismatches and severity;
- any implemented low-risk vLLM-aligned adaptation;
- recommendation for TARGET 10.2.

## Done Criteria

Done when one of these is true:

- mini/vLLM communication paths match at owner/count/bytes level, so backend
  experiments are justified;
- a material path mismatch is identified and either fixed or assigned as the
  next concrete adaptation;
- vLLM runtime instrumentation is blocked for a concrete reason, but a
  source-derived table is detailed enough to guide TARGET 10.2 safely and the
  static rows are clearly marked.

## Stop Rules

Stop and report instead of broadening if:

- vLLM path cannot be mapped to specific owners;
- communication-count differences are noisy because workloads are not comparable;
- a proposed adaptation would change model semantics or precision without a
  correctness gate;
- the work drifts into PyNCCL/NCCL tuning before path parity is understood.

## Non-Goals

- PyNCCL, symmetric-memory, or NCCL backend tuning.
- Low-precision changes.
- Prefix-cache ownership changes.
- Attention kernel rewrites.

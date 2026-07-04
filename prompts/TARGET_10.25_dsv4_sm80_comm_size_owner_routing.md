# TARGET 10.25: DSV4 SM80 Communication Size/Owner Routing

## Status

Run after TARGET 10.2.

TARGET 10.2 found that communication backend choice is size-sensitive:

- PyNCCL direct wins some isolated large-collective microbench rows, but the
  no-weight historical trace was neutral;
- PyNCCL symmetric buffer is bad for 128 MiB hidden all-reduces because
  copy-in/copy-out dominates;
- PyNCCL symmetric buffer is competitive for about 20 MiB serving-wave hidden
  all-reduces;
- the best full-model single-run candidate was:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

This target decides whether that candidate should remain opt-in, be promoted as
a simple global PyNCCL threshold, or be replaced by a more explicit
per-owner/per-size routing layer.

## Goal

Build a repeat-stable communication routing decision for DeepSeek V4 Flash on
TP8/A100/sm80, reusing mini's existing PyNCCL/symmetric-memory interfaces where
possible.

The key question is:

```text
Should mini route communication by owner/op/shape/dtype instead of using a
single global Torch/NCCL or PyNCCL backend?
```

The expected answer may be mixed:

- Torch/NCCL for large prefill-style hidden all-reduces;
- PyNCCL symmetric-buffer path for smaller fixed-shape decode all-reduces;
- PyNCCL direct for selected all-gather or large all-reduce only if repeat
  evidence supports it;
- fallback to Torch/NCCL for fp32 or uncommon shapes unless proven otherwise.

## Required Inputs

Read first:

- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md`
- `prompts/TARGET_10.2_dsv4_sm80_comm_stack_backend_experiments.md`
- `performance_milestones/target10_moe_reduce_bf16_parity/README.md`
- `performance_milestones/target10_comm_stack_backend_experiments/README.md`

Mini references:

- `python/minisgl/distributed/impl.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`
- `python/minisgl/env.py`
- `python/minisgl/engine/engine.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `performance_milestones/target10_comm_stack_backend_experiments/scripts/tp8_comm_backend_probe.py`

vLLM references, only for comparison and future custom-communicator planning:

- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/all_reduce_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/cuda_wrapper.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/pynccl.py`

## Existing Interface To Reuse

Do not start by rewriting PyNCCL.

Mini already has:

- `PyNCCLDistributedImpl` in `python/minisgl/distributed/impl.py`;
- `init_pynccl(..., max_size_bytes=...)` in `python/minisgl/kernel/pynccl.py`;
- `MINISGL_PYNCCL_MAX_BUFFER_SIZE` / `ENV.PYNCCL_MAX_BUFFER_SIZE`;
- `pynccl.cu` `NCCLWrapper::all_reduce`, which already routes by size:
  - `size_bytes <= m_max_bytes`: copy input to internal symmetric buffer,
    all-reduce in-place on the buffer, copy back;
  - `size_bytes > m_max_bytes`: direct in-place `ncclAllReduce` on the input;
- `pynccl.cu` `NCCLWrapper::all_gather`, which writes directly to the output
  tensor and does not use the symmetric buffer;
- `DistributedCommunicator` labels, stats, dtype/shape bytes accounting, and
  owner timing labels.

What mini does not yet have:

- a label/shape-aware selector between Torch/NCCL and PyNCCL;
- a first-class route summary that records which backend handled each owner;
- separate per-owner route policy for all-reduce versus all-gather.

The preferred implementation path is therefore:

1. reuse the existing PyNCCL communicator and `m_max_bytes` threshold;
2. add routing at the `DistributedCommunicator` layer, where `label`, op,
   dtype, shape, and bytes are visible;
3. keep a simple global threshold candidate as a baseline;
4. only touch `pynccl.cu` if the routing experiment proves the current interface
   is insufficient.

## Baseline

Use the fixed TARGET 10.15 dtype path:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Do not change model precision, MoE backend, attention kernel, or prefix-cache
ownership in this target.

## Work Plan

### 1. Repeat-Stable Gate For Existing Threshold32m

Before adding new routing code, validate the best TARGET 10.2 candidate:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

Run at least:

- two repeats of `historical_4096_1024_bs4`;
- two repeats of `serving_mixed_112req_wave16`;
- one `prefix_multi_112req_wave16` health run;
- text smoke.

Compare against a same-run Torch/NCCL fixed-BF16 baseline. Record graph
replay/eager and communication stats.

If the repeat gate is neutral or negative, do not spend the whole target on
routing implementation. Report that the global threshold candidate is not
stable enough and keep PyNCCL opt-in.

### 2. Design A Minimal Route Policy

If threshold32m remains positive, design a small routing policy that can express
at least these candidates:

- `torch_all`: current Torch/NCCL default;
- `pynccl_threshold32m`: current global PyNCCL candidate;
- `route_small_hidden_to_pynccl`: Torch/NCCL for large hidden all-reduces,
  PyNCCL threshold path for smaller hidden all-reduces;
- `route_hidden_to_pynccl`: PyNCCL threshold path for BF16 hidden all-reduces,
  Torch/NCCL for everything else;
- optional `route_gather_to_pynccl`: PyNCCL for `lm_head_all_gather` only if
  allocation and graph behavior remain clean.

Start with the owners from TARGET 10.1/10.15:

- `dsv4.attn.wo_b.row_parallel_projection_all_reduce`;
- `dsv4.v1_moe_reduce_once_all_reduce`;
- `dsv4.embedding_all_reduce`;
- `dsv4.lm_head_all_gather`.

The route decision should use op, label, dtype, and input bytes. It should not
depend on Python-side timing or request contents during graph replay.

Keep the route opt-in, for example:

```text
MINISGL_COMM_ROUTE_POLICY=dsv4_sm80_size_owner_v1
```

The exact name can follow local env style.

### 3. Microbench And No-Weight Replay

Reuse or extend the TARGET 10.2 probe script.

For every route candidate, run:

- pure communication microbench on the exact owner shapes;
- no-weight trace replay for:
  - `historical_4096_128_bs4`;
  - `historical_4096_1024_bs4` if cheap;
  - `serving_mixed_112req_wave16`;
  - one prefix-like mixed-shape trace if cheap.

Record:

- median/P95 latency;
- graph capture/replay compatibility;
- D2D copy bytes;
- selected backend per owner/shape;
- BF16/fp32 correctness against Torch/NCCL.

Reject a route before full-model testing if no-weight replay is neutral or
negative compared with the existing threshold32m candidate.

### 4. Implement Minimal Routing Only For Surviving Candidates

If a route survives the cheap gates, implement the smallest production-shaped
hook.

Preferred shape:

- keep Torch/NCCL as the default backend;
- initialize PyNCCL only when routing is enabled;
- select backend inside `DistributedCommunicator.all_reduce/all_gather`, where
  label and tensor metadata are available;
- add route stats showing `label/op/dtype/shape/backend/count/bytes`;
- preserve owner timing labels;
- keep `--use-pynccl` behavior working as the existing global route.

Do not introduce a complex policy language. Hard-code one or two DSV4 sm80
policies behind env flags if that keeps the implementation clear and reversible.

### 5. Full-Model Correctness And Macro Gate

For any implemented candidate:

1. Run text smoke.
2. Run same-run macro A/B against Torch/NCCL fixed-BF16 baseline.
3. Run repeat-stable macro if first pass is positive.
4. Run owner timing/profile only after repeat-stable macro stays positive.

Minimum macro scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`.

Record graph replay/eager and route stats for every scenario.

### 6. Promotion Decision

Promote only if:

- text smoke passes;
- BF16 and fp32 correctness probes pass;
- graph replay remains zero-eager;
- route stats show the intended backend per owner/shape;
- repeat-stable macro improvement is at least `2%` E2E or clearly improves the
  decode-forward envelope;
- the route policy is simple enough to maintain;
- fallback to Torch/NCCL is one env/config change.

If the result is positive but below the promotion bar, keep the route as opt-in
and document when to use it.

## Deliverables

Write:

```text
performance_milestones/target10_comm_size_owner_routing/README.md
```

Include:

- repeat-stable threshold32m gate result;
- existing PyNCCL interface reuse summary;
- route policy design;
- pure communication and no-weight replay comparison;
- implementation summary, if any;
- route stats by owner/shape/dtype/backend;
- D2D copy accounting;
- graph-capture compatibility;
- text smoke and correctness result;
- macro A/B and repeat-stability table;
- promote/keep-opt-in/reject decision;
- recommendation for either TARGET 10.3 overlap/custom communicator work or a
  vLLM custom all-reduce port target.

## Done Criteria

Done when one of these is true:

- the global PyNCCL threshold32m route is repeat-stable enough to promote or
  keep as a documented opt-in;
- a per-owner/per-size route is implemented and promoted or kept opt-in;
- cheap gates show routing is not worth implementing, and PyNCCL remains opt-in;
- a vLLM custom communicator port becomes the clear next target.

## Stop Rules

Stop and report instead of broadening if:

- the threshold32m repeat gate is neutral/negative;
- no-weight replay contradicts isolated microbench wins;
- route implementation breaks CUDA graph replay;
- route implementation makes communicator lifecycle fragile;
- the work drifts into FP8/INT8, attention kernels, prefix-cache ownership, or
  model-parallel partitioning;
- a custom P2P/all-reduce implementation becomes necessary. That should be a
  separate target.

## Non-Goals

- New low-precision model paths.
- Attention kernel work.
- Prefix-cache/SWA ownership changes.
- Rewriting PyNCCL from scratch.
- Porting vLLM custom all-reduce in this target.

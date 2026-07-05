# TARGET 10.2: DSV4 SM80 Communication Stack Backend Experiments

## Status

Run after TARGET 10.1 communication path parity and TARGET 10.15 MoE
reduce-once BF16 parity.

Do not start this target until mini/vLLM communication owner/count/bytes
parity is known, or until TARGET 10.1 identifies a deliberate path difference
that should remain. Also do not start while the MoE reduce-once fp32-vs-BF16
dtype mismatch from TARGET 10.1 remains untested.

## Goal

Evaluate whether the communication stack itself is a bottleneck after path
parity:

- PyTorch distributed/NCCL;
- mini PyNCCL;
- mini PyNCCL symmetric-memory workspace path;
- vLLM custom all-reduce and quick all-reduce ideas;
- CUDA P2P / CUDA IPC as low-level building blocks for custom collectives;
- reduce-scatter or all-gather boundary changes only if TARGET 10.1 shows that
  vLLM uses a different logical communication boundary;
- per-owner or per-size backend routing;
- vLLM custom/symmetric communicator ideas that can be adapted cleanly.

The goal is to answer:

```text
Which communication owners and tensor sizes, if any, should use a backend other
than the current default?
```

The communication stack must remain functionally correct for both BF16 and
fp32, because fp32 can still appear at fallback, logits/sampler, debug, or
future correctness boundaries.  The optimization target is narrower: after
TARGET 10.15, the hot decode-forward communication path should be treated as
BF16 unless a remaining fp32 collective is explicitly justified.

## Required Inputs

Read first:

- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/archive/target10/TARGET_10.1_dsv4_sm80_comm_path_parity_vllm.md`
- `prompts/archive/target10/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md`
- `performance_milestones/target10_comm_path_parity_vllm/README.md`
- `performance_milestones/target10_moe_reduce_bf16_parity/README.md`

Mini references:

- `python/minisgl/distributed/impl.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`
- `python/minisgl/env.py`
- `python/minisgl/engine/engine.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

vLLM references:

- source tree: `/workspace/vllm-dsv4-docker`
- virtual environment: `/workspace/venvs/vllm-dsv4`
- Python executable: `/workspace/venvs/vllm-dsv4/bin/python`
- TARGET 07 runtime templates:
  `performance_milestones/target07_vllm_gap/scripts/run_vllm_4096x1024_bs4.sh`
  and
  `performance_milestones/target07_vllm_gap/scripts/nsys_vllm_4096x128_bs4_fair.sh`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/quick_all_reduce.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/symm_mem.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/all_reduce_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/cuda_wrapper.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/pynccl.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`

## Baseline

Use the TARGET 08 prefix baseline and the TARGET 10.1 parity workload set:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Keep a non-prefix `dsv4_sm80_a100_victory` control only if the experiment is
clearly unrelated to prefix cache.

If TARGET 10.15 promotes BF16 MoE reduce, use that promoted path as the default
baseline for this target. If it keeps BF16 reduce as opt-in, run backend
experiments on the selected fixed dtype and clearly state which dtype is used.
Also carry forward the remaining-fp32-collective audit from TARGET 10.15, so
PyNCCL/custom-communicator experiments can focus on the final hot dtype set
instead of supporting accidental fp32 traffic.

Minimum fixed dtype set for this target:

- hot BF16 all-reduce shapes:
  - `dsv4.attn.wo_b.row_parallel_projection_all_reduce`;
  - `dsv4.v1_moe_reduce_once_all_reduce` with
    `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`;
  - `dsv4.embedding_all_reduce`;
- remaining fp32 all-gather shape:
  - `dsv4.lm_head_all_gather`, covered as functional/fallback traffic, not as
    the primary optimization target unless a tiny logits BF16 gather experiment
    is explicitly split out.

## Work Plan

### 1. Backend Inventory

Document the current mini stack:

- default `TorchDistributedImpl` behavior;
- `PyNCCLDistributedImpl` behavior;
- `PYNCCL_MAX_BUFFER_SIZE`;
- symmetric-memory window allocation;
- copy-in/copy-out behavior for `all_reduce`;
- direct output behavior for `all_gather`.

Document relevant vLLM stack ideas:

- custom all-reduce;
- quick all-reduce;
- symmetric-memory communicator;
- PyNccl communicator;
- CUDA IPC handle exchange and P2P access checks;
- reduce-scatter support;
- any size thresholds or graph-capture behavior.

Also document which ideas are complete collective backends versus lower-level
building blocks.  For example, CUDA P2P/IPC can enable a custom communicator,
but raw peer copies alone are not a drop-in replacement for TP8 all-reduce.

### 2. Isolated Microbench

Build or reuse a small TP8 microbench that replays the actual TARGET 10.1 and
TARGET 10.15 collective shapes by owner label.  Start with pure communication
functions, without model weights and without DSV4 forward.

For each owner/shape/dtype, compare:

- PyTorch/NCCL;
- mini PyNCCL direct path when tensor exceeds symmetric buffer;
- mini PyNCCL symmetric-buffer path when tensor fits;
- optional vLLM-inspired custom all-reduce or quick all-reduce path if feasible;
- optional CUDA P2P/IPC peer-copy microbench for the same sizes, clearly marked
  as a building-block measurement rather than a full collective unless a full
  all-reduce/all-gather implementation exists;
- reduce-scatter/all-gather decomposition only when TARGET 10.1 identifies a
  vLLM-aligned logical-boundary difference.

Measure:

- median/P95 latency;
- achieved bandwidth;
- mean and tail latency across enough iterations to expose launch/capture
  overhead;
- D2D copy bytes/time when symmetric buffer is used;
- IPC/P2P setup cost, peer-access availability, and steady-state peer-copy
  latency/bandwidth if P2P is tested;
- CUDA stream behavior;
- graph-capture compatibility for the exact candidate path;
- correctness by comparing against PyTorch/NCCL output for BF16 and fp32.

Treat mini PyNCCL symmetric memory as the primary route to investigate first,
because mini already has a prototype.  Still include small microbench evidence
for other routes when cheap:

- plain PyTorch/NCCL baseline;
- mini PyNCCL direct path;
- mini PyNCCL symmetric-buffer path;
- vLLM custom/quick all-reduce feasibility from source and, if low effort, a
  shape-compatible prototype;
- CUDA P2P/IPC peer-copy only as a building-block measurement.

### 3. No-Weight Trace Replay

Before loading DeepSeek weights, build a no-weight TP8 replay that issues the
same sequence of collectives as the target scenarios using synthetic tensors.

The replay should cover at least:

- `historical_4096_128_bs4`;
- one wave-shaped scenario such as `serving_mixed_112req_wave16`.

Replay variants:

- eager loop;
- CUDA graph capture/replay when the backend claims graph compatibility;
- per-owner or thresholded routing candidates.

The goal is to validate:

- whether microbench wins survive realistic ordering and repetition;
- whether graph capture works without model noise;
- whether symmetric-memory copy-in/copy-out overhead compounds across the full
  trace;
- whether fp32 fallback traffic still works.

### 4. Partial Runtime Probe

Classify owners:

- large all-reduce likely better on plain NCCL;
- small or repeated all-reduce possibly better on symmetric-memory/custom path;
- mid-sized fixed-shape all-reduce may be a candidate for vLLM-style custom AR
  or quick AR if P2P/IPC checks pass;
- all-gather likely not helped by the current mini symmetric buffer;
- raw CUDA P2P is not promotable by itself unless wrapped into a correct
  collective with synchronization and graph-capture behavior understood;
- backend-neutral.

Do not route all collectives to PyNCCL blindly.  If a backend only wins for one
owner/size range, make the opt-in per-owner or thresholded.

If the no-weight replay identifies a promising backend, integrate it behind a
minimal opt-in and run a partial runtime probe before full macro:

- initialize the real distributed/communicator stack;
- avoid loading model weights when possible;
- use the same env flags and graph capture path that the full engine would use;
- run output correctness against PyTorch/NCCL for synthetic tensors.

This step should catch ABI, lifecycle, graph-capture, and stream-ordering
issues before the expensive full model run.

### 5. Full Model Correctness And TP8 Macro A/B

Run repeat-stable macro comparisons:

- current default;
- all PyNCCL if already supported;
- candidate per-owner/threshold routing;
- optional vLLM-inspired custom/quick all-reduce path;
- optional reduce-scatter/all-gather boundary adaptation if TARGET 10.1 proved
  that this is a path-parity fix rather than pure backend tuning.

Minimum scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16` for prefix health.

Before interpreting performance, run text smoke for any candidate that changes
the full engine communication path.  Then record graph replay/eager status,
communication stats, and owner timing/profile evidence for any promoted
candidate.

### 6. Promotion Decision

Promote only if the macro gain is repeat-stable and the routing rule is simple.
Otherwise keep as opt-in or reject.

## Deliverables

Write:

```text
performance_milestones/target10_comm_stack_backend_experiments/README.md
```

Include:

- backend inventory;
- pure communication microbench table by owner/shape/dtype;
- no-weight trace replay results;
- partial runtime probe result if implemented;
- D2D copy overhead accounting;
- graph-capture compatibility table;
- BF16 and fp32 correctness comparison against PyTorch/NCCL;
- CUDA P2P/IPC feasibility and peer-access table if tested;
- vLLM custom/quick all-reduce applicability summary;
- TP8 macro A/B table;
- graph replay/eager table;
- text smoke result for any candidate;
- promote/keep-opt-in/reject decision.

## Done Criteria

Done when one of these is true:

- a per-owner or thresholded communication backend improves TP8 macro enough to
  promote;
- PyNCCL/symmetric memory is proven neutral or negative for current owners and
  rejected;
- pure microbench or no-weight trace replay shows that a candidate is not worth
  integrating into the full model;
- a vLLM custom communicator idea is identified as promising but requires a
  separate porting target.
- CUDA P2P/IPC is classified as useful or not useful for the current owner
  shapes, without pretending it is a standalone collective backend.

## Stop Rules

Stop and report instead of broadening if:

- path parity from TARGET 10.1 is missing;
- TARGET 10.15 did not resolve or intentionally freeze the MoE reduce dtype
  boundary;
- graph replay breaks and cannot be restored quickly;
- PyNCCL wins only by measurement noise;
- D2D copy overhead cancels symmetric-memory benefit;
- P2P/IPC setup, synchronization, topology, or graph-capture constraints make a
  custom path unsafe or non-competitive;
- implementation drifts into low precision, attention kernels, or prefix-cache
  ownership.

## Non-Goals

- Changing model parallel partitioning.
- Changing precision.
- Rewriting attention kernels.
- Prefix-cache redesign.
- Solving unrelated graph memory-pool capacity issues.

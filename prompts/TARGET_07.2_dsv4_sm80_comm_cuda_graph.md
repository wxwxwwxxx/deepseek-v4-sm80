# TARGET 07.2: DSV4 sm80 Communication and CUDA Graph

## Goal

Reduce mini-sglang's DeepSeek V4 decode overhead by fixing communication
observability, evaluating the available PyNCCL path, and enabling decode CUDA
graph capture for stable DSV4 batch sizes.

This target is complete when the 4096/128/batch4 short profile shows a large
drop in kernel/runtime event count and clear evidence of CUDA graph replay or a
documented reason why graph capture is still blocked.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- Fair comparison target:
  `prompts/TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md`
- mini graph runner: `python/minisgl/engine/graph.py`
- DSV4 graph disable gate: `python/minisgl/engine/engine.py`
- DSV4 attention backend: `python/minisgl/attention/deepseek_v4.py`
- DSV4 model communication call sites:
  `python/minisgl/models/deepseek_v4.py`
- communication abstraction: `python/minisgl/distributed/impl.py`
- PyNCCL wrapper: `python/minisgl/kernel/pynccl.py`,
  `python/minisgl/kernel/csrc/src/pynccl.cu`

vLLM references:

- CUDA graph dispatcher:
  `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- graph-aware communication wrapper:
  `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`
- custom all-reduce:
  `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`
- all-reduce utility thresholds:
  `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/all_reduce_utils.py`

## Current Known Blockers

- `python/minisgl/engine/engine.py` currently disables CUDA graph for
  DeepSeek V4 by overriding `cuda_graph_bs=[]` and `cuda_graph_max_bs=0`.
- Existing mini DSV4 profile has no CUDA graph replay evidence.
- TARGET 06 forced `use_pynccl=false` because PyNCCL reached DSV4 forward but
  failed around `lm_head.linear()` all-gather.
- `python/minisgl/kernel/csrc/src/pynccl.cu` currently maps fp16 and bf16 NCCL
  dtypes but not fp32, while lm_head logits use fp32.
- mini V1 nsys shows NCCL all-reduce as the largest category and tens of
  millions of CUDA launches.

## Plan

1. Add communication observability.
   - Extend `DistributedCommunicator` with an optional semantic label.
   - Label DSV4 call sites at minimum:
     - embedding all-reduce;
     - attention/row-parallel projection all-reduce;
     - routed expert all-reduce;
     - shared expert all-reduce;
     - V1 reduce-once MoE all-reduce;
     - lm_head all-gather.
   - Record count, dtype, shape, bytes, and total calls per label in the
     benchmark report.

2. Repair and evaluate PyNCCL.
   - Add fp32 support to the local PyNCCL dtype map.
   - Add direct communication tests for bf16/fp16/fp32 all-reduce and
     all-gather across TP ranks.
   - Run DSV4 TP8 text smoke with PyNCCL enabled and page size 256.
   - Add a benchmark variant such as `v1_moe_pynccl` only after correctness
     passes.

3. Enable DSV4 decode CUDA graph safely.
   - Replace the unconditional DSV4 graph disable gate with an explicit flag or
     variant-controlled allowlist.
   - Start with decode-only graph capture for batch sizes `[1,2,4]`.
   - Keep prefill eager.
   - Ensure DSV4 attention metadata, page table references, positions,
     input ids, and output locations are stable graph inputs or copied into
     capture buffers before replay.
   - If capture fails, record the exact unsupported op or mutable metadata
     blocker in this target.

4. Compare with vLLM communication design.
   - Inspect whether vLLM's custom all-reduce is usable on the local single-node
     A100 topology.
   - Prefer the existing mini PyNCCL/symmetric-memory path if it closes most of
     the gap.
   - Port or adapt vLLM custom all-reduce only if PyNCCL remains a measured
     bottleneck after graph capture.

5. Measure impact.
   - Run 4096/128/batch4 nsys before/after for:
     - `v1_moe`;
     - `v1_moe_graph`;
     - `v1_moe_pynccl`;
     - combined graph + PyNCCL if both pass.
   - Run 4096/1024/batch4 macro benchmark for the best exact variant.

## Done Criteria

- Communication counters identify which DSV4 semantic call sites dominate
  all-reduce/all-gather count and bytes.
- PyNCCL either passes TP8 DSV4 text smoke or has a precise recorded blocker.
- DSV4 decode CUDA graph either replays for `[1,2,4]` or has a precise recorded
  blocker.
- The best exact variant has a new 4096/128 nsys report and a 4096/1024 macro
  result.
- The target updates TARGET 07 with the next bottleneck after communication and
  graph work.

## Non-Goals

- Do not implement MoE exact V2 in this target.
- Do not full-compile the whole DSV4 model.
- Do not add vLLM as a runtime dependency.
- Do not promote PyNCCL or custom all-reduce by default without TP8 text smoke
  and benchmark evidence.

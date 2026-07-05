# TARGET 10: DSV4 SM80 Decode-Forward Communication Roadmap

## Status

Recommended next family after TARGET 08.

TARGET 08 closed the prefix-cache milestone with the tag
`dsv4-sm80-prefix-routeb-lifetime-baseline`.  The promoted prefix preset is:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

TARGET 08.30 showed that prefix metadata/runtime is no longer the first
bottleneck.  Owner timing now points back to decode forward and communication
owners, especially:

- attention `wo_b` row-parallel all-reduce;
- MoE reduce-once all-reduce;
- embedding all-reduce;
- lm-head all-gather.

## Goal

Close or explain the remaining decode-forward communication gap without
guessing.

The guiding rule is:

```text
First prove whether mini's communication graph matches vLLM/SGLang at the
owner/count/bytes/shape level.  Only then tune the communication backend.
```

## Split Plan

Run in this order:

| Stage | Prompt | Status | Purpose |
| --- | --- | --- | --- |
| TARGET 10.1 | `prompts/TARGET_10.1_dsv4_sm80_comm_path_parity_vllm.md` | completed | Compare mini and vLLM communication paths: owner boundaries, collective count, bytes, dtype, shapes, metadata/runtime overhead, and graph placement. |
| TARGET 10.15 | `prompts/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md` | completed | Isolated the high-severity MoE reduce-once dtype/bytes mismatch found by 10.1; BF16 reduce is implemented and kept as explicit opt-in. |
| TARGET 10.2 | `prompts/TARGET_10.2_dsv4_sm80_comm_stack_backend_experiments.md` | completed | Tested communication backends with micro-first and no-weight replay gates; best candidate was PyNCCL threshold32m opt-in, but not repeat-stable enough to promote. |
| TARGET 10.25 | `prompts/TARGET_10.25_dsv4_sm80_comm_size_owner_routing.md` | completed | Validated PyNCCL threshold32m as a positive opt-in; explicit per-owner/per-size routing did not beat the global threshold in no-weight replay. |
| TARGET 10.26 | `prompts/TARGET_10.26_dsv4_sm80_pynccl_threshold32m_promotion_gate.md` | completed | PyNCCL threshold32m became the recommended opt-in: repeat-stable macro wins and zero-eager graph replay, but default promotion was blocked by an `lm_head_all_gather` owner-timing anomaly and a full serving Nsight capture without CUDA activity. |
| TARGET 10.27 | `prompts/TARGET_10.27_dsv4_sm80_pynccl_default_promotion_blockers.md` | completed | Explained the `lm_head_all_gather` owner-timing spike as a one-time non-captured first all-gather cost, obtained rank-scoped full-model Nsight traces with CUDA activity, and default-promoted PyNCCL threshold32m for A100/sm80 DSV4. |
| TARGET 10.3 | future conditional | not written yet | If 10.27 promotes or rejects threshold32m but communication remains material, test overlap, NCCL grouping, stream scheduling, or fused compute+collective boundaries. |

Do not start broad attention-kernel work inside TARGET 10 unless a fresh profile
shows attention compute, not communication, is the top remaining owner.

## Current Mini Communication Surface

Mini has a narrow communication abstraction at:

```text
python/minisgl/distributed/impl.py
```

Current owner labels include:

- `dsv4.embedding_all_reduce`;
- `dsv4.attn.wo_b.row_parallel_projection_all_reduce`;
- `dsv4.v1_moe_reduce_once_all_reduce`;
- `dsv4.routed_expert_all_reduce`;
- `dsv4.shared_expert_all_reduce`;
- `dsv4.lm_head_all_gather`.

Important mini model references:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/layers/embedding.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`

The communicator already records per-label count/bytes/shape summaries through
`snapshot_communication_stats()`.  TARGET 10.1 should make this data comparable
to vLLM before TARGET 10.2 changes the backend.

## vLLM References

Use vLLM as the first communication-path reference:

- source tree: `/workspace/vllm-dsv4-docker`
- virtual environment: `/workspace/venvs/vllm-dsv4`
- Python executable: `/workspace/venvs/vllm-dsv4/bin/python`
- mini uses the system interpreter from `/workspace/mini-sglang`

vLLM was run extensively during TARGET 07.  Reuse those scripts and runtime
settings before assuming that vLLM cannot run in this container:

- `performance_milestones/target07_vllm_gap/README.md`
- `performance_milestones/target07_vllm_gap/scripts/run_vllm_4096x1024_bs4.sh`
- `performance_milestones/target07_vllm_gap/scripts/nsys_vllm_4096x128_bs4_fair.sh`
- `performance_milestones/vllm/scripts/vllm_env.sh`

Important source files:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/logits_processor.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/communication_op.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/`

Useful vLLM design surfaces to inspect:

- `RowParallelLinear` all-reduce behavior;
- `VocabParallelEmbedding` all-reduce behavior;
- logits processor all-gather/gather behavior;
- FusedMoE runner reductions;
- `get_tp_group().all_reduce/all_gather/reduce_scatter`;
- custom all-reduce and symmetric-memory communicators.

## Measurement Baseline

Use the TARGET 08 prefix baseline unless a target explicitly says otherwise:

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

Minimum scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16` when checking that prefix functionality remains
  healthy.

## Decision Rules

### Path Parity First

If TARGET 10.1 finds a mini/vLLM communication-count or communication-volume
difference for the same logical workload, fix or explain that difference before
changing backend implementation.

Examples:

- extra mini all-reduce not present in vLLM;
- vLLM using reduce-scatter or all-gather where mini uses all-reduce;
- different logits gather strategy;
- repeated metadata/runtime work that indirectly triggers more communication;
- dtype/shape mismatch that changes communication bytes materially.

TARGET 10.1 already found one such mismatch: mini's MoE reduce-once currently
uses fp32 while vLLM's SM80 source path indicates a BF16 hidden-state reduce.
TARGET 10.15 handles this before backend tuning.

### Backend Experiments Second

Only after path parity is understood and the MoE reduce-once dtype mismatch
from TARGET 10.1 is isolated should TARGET 10.2 test:

- PyTorch distributed/NCCL;
- mini PyNCCL;
- mini PyNCCL symmetric-memory workspace path;
- vLLM custom/quick all-reduce ideas;
- CUDA P2P/IPC as a low-level building block for custom communicators;
- per-owner or per-size backend routing;
- vLLM custom/symmetric communicator ideas that can be adapted cleanly.

TARGET 10.2 should be micro-first: pure communication functions, then
no-weight trace replay of the real owner sequence, then full-model
correctness/macro only for candidates that survive the cheaper gates.

Do not assume symmetric memory wins.  Mini's current PyNCCL all-reduce may copy
into an internal symmetric buffer and copy back; those D2D copies must be
counted against any NCCL improvement.  Do not assume raw CUDA P2P is a complete
collective either: it is mainly a building block for fixed-shape custom
all-reduce/all-gather paths, and must be compared against NCCL on the exact
owner shapes from TARGET 10.1.

### Defer Attention Compute

Attention-kernel work belongs later unless the post-communication profile shows
attention compute is again a top-two owner.  Earlier TARGET 07 evidence already
put mini's exact BF16 sparse decode near the comparable vLLM sparse-decode
probe, so do not start by rewriting C4A/C128A attention.

### Defer Low Precision

Do not introduce FP8 KV, INT8 MoE, or quantized projection changes inside
TARGET 10.  Those belong in TARGET 09.

## Current Default Communication Path

TARGET 10.27 default-promoted the fixed A100/sm80 DSV4 communication path:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL enabled by default for that preset
Default DSV4 sm80 PyNCCL max buffer size: 32M
```

The 32M cap is applied by the engine for DeepSeek V4 on sm80 only when
`MINISGL_PYNCCL_MAX_BUFFER_SIZE` is not explicitly set. To roll back, set a
different `MINISGL_PYNCCL_MAX_BUFFER_SIZE` or disable PyNCCL.

## Stop Rules

Stop a TARGET 10 child thread when it has answered its bounded question.

Stop and report blocked if:

- vLLM runtime instrumentation fails for a concrete reason, and neither the
  TARGET 07 run scripts nor a source-derived fallback can build a fair
  communication-count/bytes comparison;
- mini/vLLM path differences are found but cannot be isolated to a specific
  owner or layer family;
- PyNCCL or symmetric-memory experiments are neutral/negative after a clean
  microbench and TP8 macro;
- graph replay is broken by a communication backend change;
- a backend change only helps one tiny microbench but does not move the
  measured decode envelope.

## Promotion Rules

Promote a communication change only if all are true:

- path parity is known or intentionally changed with evidence;
- text smoke passes;
- graph replay remains zero-eager for the target buckets;
- per-owner count/bytes/timing summaries are recorded;
- TP8 macro improvement is repeat-stable and at least `2%` E2E or clearly
  improves the dominant decode-forward envelope;
- fallback/rollback path remains simple.

## Non-Goals

- Changing prefix-cache ownership; that belongs to TARGET 08 history.
- Introducing low-precision behavior; that belongs to TARGET 09.
- Running attention, communication path parity, PyNCCL, graph layout, and
  overlap experiments in one thread.
- Continuing generic TARGET 07 polishing.

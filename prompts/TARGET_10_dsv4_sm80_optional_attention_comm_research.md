# TARGET 10: DSV4 SM80 Decode-Forward Communication Roadmap

## Status

Closed communication milestone after TARGET 08.

TARGET 08 closed the prefix-cache milestone with the tag
`dsv4-sm80-prefix-routeb-lifetime-baseline`.  The promoted prefix preset is:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

TARGET 08.30 showed that prefix metadata/runtime was no longer the first
bottleneck.  Owner timing pointed back to decode forward and communication
owners, especially:

- attention `wo_b` row-parallel all-reduce;
- MoE reduce-once all-reduce;
- embedding all-reduce;
- lm-head all-gather.

## Goal

Close or explain the remaining decode-forward communication gap without
guessing.  TARGET 10 is now closed: TARGET 10.27 default-promoted the PyNCCL
threshold32m path for DeepSeek V4 Flash TP8/A100/sm80.

The guiding rule is:

```text
First prove whether mini's communication graph matches vLLM/SGLang at the
owner/count/bytes/shape level.  Only then tune the communication backend.
```

## Split Plan

Run in this order:

| Stage | Prompt | Status | Purpose |
| --- | --- | --- | --- |
| TARGET 10.1 | `prompts/archive/target10/TARGET_10.1_dsv4_sm80_comm_path_parity_vllm.md` | completed | Compared mini and vLLM communication paths: owner boundaries, collective count, bytes, dtype, shapes, metadata/runtime overhead, and graph placement. |
| TARGET 10.15 | `prompts/archive/target10/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md` | completed | Isolated the high-severity MoE reduce-once dtype/bytes mismatch found by 10.1; BF16 reduce was implemented as an explicit opt-in. |
| TARGET 10.2 | `prompts/archive/target10/TARGET_10.2_dsv4_sm80_comm_stack_backend_experiments.md` | completed | Tested communication backends with micro-first and no-weight replay gates; best candidate was PyNCCL threshold32m. |
| TARGET 10.25 | `prompts/archive/target10/TARGET_10.25_dsv4_sm80_comm_size_owner_routing.md` | completed | Validated PyNCCL threshold32m as positive; explicit per-owner/per-size routing did not beat the global threshold in no-weight replay. |
| TARGET 10.26 | `prompts/archive/target10/TARGET_10.26_dsv4_sm80_pynccl_threshold32m_promotion_gate.md` | completed | PyNCCL threshold32m became the recommended opt-in: repeat-stable macro wins and zero-eager graph replay, with two default-promotion blockers remaining. |
| TARGET 10.27 | `prompts/archive/target10/TARGET_10.27_dsv4_sm80_pynccl_default_promotion_blockers.md` | completed | Explained the `lm_head_all_gather` owner-timing spike as a one-time non-captured first all-gather cost, obtained rank-scoped full-model Nsight traces with CUDA activity, and default-promoted PyNCCL threshold32m for A100/sm80 DSV4. |

Future communication work should start from a fresh post-promotion profile.  Do
not open TARGET 10.3-style overlap, NCCL grouping, stream scheduling, or fused
compute+collective work unless communication remains a top bottleneck after the
TARGET 10.27 default path is in place.

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

The communicator records per-label count/bytes/shape summaries through
`snapshot_communication_stats()`.  TARGET 10.1 used this data to compare mini
and vLLM before TARGET 10.2 changed the backend.

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

TARGET 10 started from the TARGET 08 prefix baseline:

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

## Closeout Decisions

### Path Parity

TARGET 10.1 found that mini and vLLM communication owner boundaries largely
matched, but mini reduced the combined MoE output in fp32 while vLLM's SM80
source path indicated a BF16 hidden-state reduce.

Examples:

- extra mini all-reduce not present in vLLM;
- vLLM using reduce-scatter or all-gather where mini uses all-reduce;
- different logits gather strategy;
- repeated metadata/runtime work that indirectly triggers more communication;
- dtype/shape mismatch that changes communication bytes materially.

TARGET 10.15 implemented the BF16 MoE reduce path and later TARGET 10 work used
that fixed candidate as the communication baseline.

### Backend Experiments

After the MoE reduce-once dtype mismatch was isolated, TARGET 10.2 tested:

- PyTorch distributed/NCCL;
- mini PyNCCL;
- mini PyNCCL symmetric-memory workspace path;
- vLLM custom/quick all-reduce ideas;
- CUDA P2P/IPC as a low-level building block for custom communicators;
- per-owner or per-size backend routing;
- vLLM custom/symmetric communicator ideas that can be adapted cleanly.

TARGET 10.2 was micro-first: pure communication functions, no-weight trace
replay of the real owner sequence, then full-model correctness/macro for
surviving candidates.

Do not assume symmetric memory wins.  Mini's current PyNCCL all-reduce may copy
into an internal symmetric buffer and copy back; those D2D copies must be
counted against any NCCL improvement.  Do not assume raw CUDA P2P is a complete
collective either: it is mainly a building block for fixed-shape custom
all-reduce/all-gather paths, and must be compared against NCCL on the exact
owner shapes from TARGET 10.1.

### Attention Compute

Attention-kernel work remains deferred unless a post-communication profile shows
attention compute is again a top-two owner.  Earlier TARGET 07 evidence already
put mini's exact BF16 sparse decode near the comparable vLLM sparse-decode
probe, so do not start by rewriting C4A/C128A attention.

### Low Precision

TARGET 10 did not introduce FP8 KV, INT8 MoE, or quantized projection changes.
Those belong in TARGET 09.

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

## Opt-In Cleanup

Merged/promoted:

- PyNCCL is the default communication backend for the promoted A100/sm80 DSV4
  preset.
- DeepSeek V4 on sm80 defaults PyNCCL max buffer size to `32M` when PyNCCL is
  enabled and `MINISGL_PYNCCL_MAX_BUFFER_SIZE` is unset.
- The promoted benchmark/text-smoke preset
  `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16` includes
  `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1` and `use_pynccl=True`.

Kept as rollback or comparison surfaces:

- `MINISGL_PYNCCL_MAX_BUFFER_SIZE` remains an explicit threshold override.
- `--disable-pynccl` remains the serving rollback switch.
- Non-PyNCCL and pre-TARGET-10 presets remain useful for historical comparison
  and bisecting regressions, but should not be used as the main performance
  baseline.

Not merged:

- `MINISGL_DSV4_SM80_MOE_REDUCE_BF16` is not folded into the older
  `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE` whitelist.  It is part of the
  promoted TARGET 10 preset, but keeping it explicit there avoids silently
  changing older TARGET 07/08 artifact aliases.  Fold it into the broader
  bundle only after a future post-promotion sweep intentionally retires those
  older aliases as performance baselines.
- Explicit per-owner/per-size communication routing, because TARGET 10.25 did
  not beat the global 32M threshold in the cheap gate.
- vLLM custom all-reduce, CUDA P2P/IPC collectives, NCCL grouping, and overlap
  experiments, because TARGET 10.27 already default-promoted the simpler path
  and future communication work should be profile-driven.
- Low-precision communication/model changes, which belong to TARGET 09.

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

# TARGET 10.15: DSV4 SM80 MoE Reduce-Once BF16 Parity

## Status

Run after TARGET 10.1 and before TARGET 10.2.

This target exists because TARGET 10.1 found one high-severity communication
path mismatch that must be isolated before communication backend experiments:
mini reduces the combined MoE output as `float32`, while the vLLM SM80 source
path indicates a BF16 hidden-state output/reduce boundary.

## Goal

Make the MoE reduce-once boundary vLLM-aligned or explain why mini must keep
the current fp32 reduce.

The narrow question is:

```text
Can mini reduce the final routed+shared MoE output in BF16, preserve correctness
for DeepSeek V4 Flash, cut MoE communication bytes in half, and improve or at
least not regress TP8 macro performance?
```

This target should also leave the communication stack in a cleaner state for
TARGET 10.2: identify any remaining fp32 collectives, and remove them only when
the change is vLLM-aligned, correctness-gated, and low risk.

This is not a general low-precision target. Do not introduce FP8, INT8, act
quantization, or a new MoE backend here. The primary precision boundary under
test is the final MoE reduce-once tensor, which is returned to BF16 hidden-state
dtype immediately after the reduce in the current mini path.

## Required Inputs

Read first:

- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/TARGET_10.1_dsv4_sm80_comm_path_parity_vllm.md`
- `performance_milestones/target10_comm_path_parity_vllm/README.md`

Mini references:

- `python/minisgl/models/deepseek_v4.py`
  - current non-runner path: `DSV4MoE.forward`, especially the `.float()`
    conversions before `dsv4.v1_moe_reduce_once_all_reduce`;
  - current runner path: `DSV4FusedMoERunner.finalize_routed`,
    `DSV4FusedMoERunner.apply_shared`, and
    `DSV4FusedMoERunner.maybe_reduce_final`.
- `python/minisgl/distributed/impl.py`
- `python/minisgl/env.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

vLLM references:

- source tree: `/workspace/vllm-dsv4-docker`
- virtual environment: `/workspace/venvs/vllm-dsv4`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/mxfp4.py`

Evidence from TARGET 10.1:

- mini `historical_4096_128_bs4` MoE reduce-once:
  `float32`, `[16384,4096]`, `688` calls, `172 GiB`;
- same-shape attention `wo_b` BF16 all-reduce:
  `bfloat16`, `688` calls, `86 GiB`;
- mini logits all-gather is also fp32, while vLLM's runtime probe observed
  BF16 logits all-gather. This is much smaller than MoE and should remain a
  secondary item unless the MoE change is already stable;
- vLLM backbone collectives were source-derived static fallback rows because
  torch.compile/CUDA graph/custom ops bypassed Python monkeypatch counters.

## Baseline

Use the TARGET 08 prefix baseline and TARGET 10.1 workload set:

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

Also keep a non-prefix `dsv4_sm80_a100_victory` control if it helps separate
prefix overhead from MoE reduce behavior.

## Work Plan

### 1. Reconfirm The Boundary

Document the exact mini dtype flow:

- routed expert output dtype;
- shared expert output dtype;
- local combine dtype;
- all-reduce input dtype;
- post-reduce return dtype.

Do this for both:

- the current default/victory path;
- the vLLM-runner path if enabled by the current bundle.

Also document the vLLM source-derived boundary again. If a small runtime
instrumentation probe can confirm the vLLM tensor dtype without breaking
TorchDynamo/CUDA graph capture, use it. If not, keep it as a clearly marked
source-derived inference and do not spend the whole target fighting vLLM
instrumentation.

### 2. Implement A Minimal Opt-In BF16 Reduce

Add one opt-in switch for the mini MoE reduce-once boundary, for example:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
```

The exact name can follow the existing env style in `python/minisgl/env.py`.

Candidate implementation:

- keep routed/shared local computation unchanged at first;
- before `dsv4.v1_moe_reduce_once_all_reduce`, cast the combined local output
  to the hidden-state dtype, normally `torch.bfloat16`;
- run the all-reduce on the BF16 tensor;
- return the post-reduce tensor in the same dtype expected by the surrounding
  hidden-state path.

Apply the same semantics to the runner and non-runner paths if both are live.
Do not leave one path silently using fp32 unless the report explains that it is
dead or intentionally deferred.

Keep fallback simple: setting the opt-in to `0` must restore the current fp32
reduce behavior.

### 3. Correctness Gate

Run a correctness gate before macro performance:

- text smoke with page size `256`;
- a deterministic short prompt set if an existing smoke script supports it;
- compare fp32-reduce versus BF16-reduce logits or next-token outputs on a
  small TP8 run when feasible;
- record max/mean logit drift if logits can be collected cheaply;
- record whether generated text becomes malformed, repetitive, or visibly
  degraded.

Do not promote the BF16 reduce if text smoke fails or if logit drift is much
larger than other already accepted BF16 boundaries without a clear explanation.

### 4. Communication And Profile Validation

Collect communication stats for baseline and opt-in:

- `dsv4.v1_moe_reduce_once_all_reduce` dtype;
- count;
- shape;
- bytes;
- graph replay/eager status.

Expected first-order result:

```text
MoE reduce-once bytes should drop by about 2x for the same shapes and counts.
```

Also produce a remaining-fp32-collective audit. The desired state before
TARGET 10.2 is that the hot communication path is BF16 unless a fp32 collective
has a clear semantic reason and a recorded correctness/performance tradeoff.

Then profile at least one short case, preferably `historical_4096_128_bs4`, to
check whether NCCL kernel family changes from f32 to bf16 and whether decode
owner timing improves.

### 5. Optional Secondary Logits All-Gather Check

Only after the MoE reduce BF16 path passes correctness and communication-byte
checks, optionally test a separate logits all-gather BF16 opt-in if it is simple
and vLLM-aligned:

- compute logits as the current path requires;
- cast only the communicated logits shard to BF16 before `lm_head_all_gather`;
- after all-gather, cast back to fp32 before sampling if the sampler expects
  fp32;
- compare top-1/top-k stability and text smoke against the fp32 gather path.

Keep this secondary item small. If it complicates correctness or sampling, defer
it to a separate target and finish MoE reduce BF16 first.

### 6. TP8 Macro A/B

Run repeat-stable macro comparisons:

- current baseline;
- BF16 MoE reduce opt-in.
- BF16 MoE reduce plus BF16 logits all-gather only if the secondary item is
  implemented and passes smoke.

Minimum scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`.

For expensive scenarios, it is acceptable to run the full set once, then repeat
only the most decision-relevant scenario if the signal is clear. Record this
choice explicitly.

### 7. Promotion Decision

Promote BF16 MoE reduce into the relevant victory bundle only if:

- correctness smoke passes;
- graph replay remains zero-eager for target buckets;
- MoE reduce-once communication bytes drop as expected;
- TP8 macro is repeat-stable and improves, or is neutral but removes a clear
  vLLM parity mismatch with no correctness downside;
- rollback is one env flag or a small isolated code path.

If performance is neutral or negative despite bytes dropping, keep the opt-in
and record that the current NCCL path is likely latency/scheduling-bound rather
than bandwidth-bound for this owner. TARGET 10.2 can still use the fixed dtype
as the clean backend experiment baseline.

Promote logits BF16 gather only if it is correctness-clean and either improves
performance or materially simplifies the hot communication dtype set for TARGET
10.2 without affecting sampling behavior. Otherwise keep it out of the main
bundle.

## Deliverables

Write:

```text
performance_milestones/target10_moe_reduce_bf16_parity/README.md
```

Include:

- source dtype-flow table for mini and vLLM;
- implementation summary and env flag;
- correctness smoke/logit-drift result;
- communication stats before/after;
- remaining fp32 communication audit;
- NCCL/profile summary for the short case;
- macro A/B table;
- promote/keep-opt-in/reject decision;
- recommendation for TARGET 10.2.

## Done Criteria

Done when one of these is true:

- BF16 MoE reduce is implemented, validated, and promoted or kept as an opt-in;
- correctness fails and the report explains why fp32 reduce must remain;
- bytes drop as expected but performance does not improve, and the report gives
  enough evidence for TARGET 10.2 backend experiments on the fixed dtype.

## Stop Rules

Stop and report instead of broadening if:

- the implementation drifts into FP8/INT8/act quantization;
- changes require rewriting the MoE backend rather than the final reduce
  boundary;
- graph replay breaks and cannot be restored quickly;
- correctness smoke fails;
- vLLM dtype cannot be runtime-observed but source evidence is already
  sufficient for this mini-side experiment;
- macro results are noisy and hide the communication counter outcome.

## Non-Goals

- PyNCCL/NCCL/custom all-reduce backend tuning.
- CUDA P2P/IPC experiments.
- INT8 MoE or FP8 activation/KV cache work.
- Attention kernel changes.
- Prefix-cache ownership changes.
- Rewriting the routed expert backend.

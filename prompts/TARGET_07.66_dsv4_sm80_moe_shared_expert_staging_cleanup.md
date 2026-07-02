# TARGET 07.66: DSV4 SM80 MoE/Shared-Expert Staging Cleanup

Date: 2026-07-02

## Goal

Reduce the largest remaining graph-replay `direct_copy` owner group in the
current A100/sm80 victory stack:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

This target is about the current bf16/exact serving route.  It should clean up
MoE/shared-expert dtype, projection, and materialization boundaries without
changing the intended precision policy.  Do not include an INT8 MoE route in
this target.

The target should answer three questions:

1. which MoE/shared-expert source boundaries create the measured direct-copy
   cost;
2. which of those copies are required by numerics/backend contracts and which
   are removable staging artifacts;
3. whether one narrow opt-in implementation can reduce the owner group and
   produce a measurable macro gain.

## Starting Point

Current confirmed promoted macro from TARGET 07.63:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `59.5264` | `150.2022` | `508` | `0` |
| 4096/1024/batch4 | `119.4153` | `149.1220` | `4092` | `0` |

TARGET 07.64 added `dsv4_sm80_a100_victory_metadatadeforest` as an opt-in
metadata helper, but it did not clear the profile promotion gate.  Keep it as
an ablation only; do not promote it in this target unless a fresh, separate
decision says otherwise.

TARGET 07.65 completed direct-copy owner attribution.  The promoted
4096/128/batch4 rank0 profile reported:

| Metric | Value |
| --- | ---: |
| total direct_copy | `0.737039s` |
| named owner direct_copy | `0.736794s` |
| named owner coverage | `99.97%` |
| residual | `0.000245s` |

The largest clean owner group is:

| Owner group | Kernel s | Share of direct_copy |
| --- | ---: | ---: |
| MoE/shared expert staging | `0.379204` | `51.45%` |
| attention/indexer boundary | `0.138539` | `18.80%` |
| hidden-carrier staging | `0.080656` | `10.94%` |
| MoE routed runner | `0.075389` | `10.23%` |
| sampler/logits/head | `0.048699` | `6.61%` |
| batch forward bridge | `0.013735` | `1.86%` |
| graph/replay metadata | `0.000290` | `0.04%` |

Top MoE/shared direct-copy owners:

| Direct-copy owner | Kernel s | Count | Share |
| --- | ---: | ---: | ---: |
| `dsv4.shared_experts.gate_up_proj` | `0.165751` | `26802` | `22.49%` |
| `dsv4.shared_experts.down_proj` | `0.119724` | `26835` | `16.24%` |
| `dsv4.layer*.mlp.runner.experts` | `0.053714` | `16072` | `7.29%` |
| `dsv4.layer*.mlp.runner.shared` | `0.031286` | `10722` | `4.24%` |
| `moe_shared_expert_staging.runner_finalize_to_fp32.layer*` | `0.022872` | `5354` | `3.10%` |
| `dsv4.layer*.mlp.runner.route` | `0.021675` | `5773` | `2.94%` |
| `moe_shared_expert_staging.runner_shared_to_fp32.layer*` | `0.020026` | see 07.65 full table | see 07.65 full table |
| `moe_shared_expert_staging.runner_output_to_flat_dtype.layer*` | `0.011815` | see 07.65 full table | see 07.65 full table |

The 07.64 metadata opt-in did not change the owner shape:

| Owner group | Promoted s | 07.64 opt-in s |
| --- | ---: | ---: |
| MoE/shared expert staging | `0.379204` | `0.380568` |
| attention/indexer boundary | `0.138539` | `0.136524` |
| hidden-carrier staging | `0.080656` | `0.080561` |
| MoE routed runner | `0.075389` | `0.076437` |

Therefore this target should focus on MoE/shared experts, not metadata,
scheduler bridge, or graph input staging.

## Relevant Mini Source Boundaries

Primary files:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
performance_milestones/target07_direct_copy_owner_attribution/
```

Important mini locations:

- `DSV4SharedExperts.forward`:
  `python/minisgl/models/deepseek_v4.py`
  - `shared_experts.gate_up_proj`;
  - `silu_and_mul_clamp_fallback`;
  - `shared_hidden_to_up_dtype`;
  - `shared_experts.down_proj`.
- `DSV4FusedMoERunner`:
  `python/minisgl/models/deepseek_v4.py`
  - `finalize_routed`;
  - `apply_shared`;
  - `maybe_reduce_final`;
  - final `y.to(flat.dtype).view_as(hidden_states)`.
- `DSV4Linear` cached FP8 BF16 helpers:
  `prepare_fp8_bf16_weight_cache` and `forward_fp8_cached_bf16_weight`.
  These already exist for attention projection owners and may be reusable for
  shared experts.
- `quantized_linear_ref`:
  `python/minisgl/kernel/deepseek_v4.py`.
  Inspect whether shared expert projection still goes through generic FP8
  activation quant/dequant/layout staging rather than a cached or fused path.

## vLLM Reference

Use vLLM as a design reference, but do not blindly port unrelated precision
paths.

Relevant source roots:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/layer.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/
```

Specific vLLM behavior to inspect:

- `DeepseekV4MoE` passes `shared_experts` into `FusedMoE`, so shared experts may
  be owned by the fused-MoE runner rather than by a separate mini-style Python
  finalization path.
- `runner/shared_experts.py` supports `SharedExpertsOrder`, including
  `MK_INTERNAL_OVERLAPPED` and `MULTI_STREAM_OVERLAPPED`.
- vLLM can run shared experts on an auxiliary stream for small token counts,
  or let the modular kernel own overlap/finalization when the backend supports
  it.

Record whether vLLM's sm80 DSV4 path uses overlap, cached/repacked weights,
different finalization precision, or a different output layout.  If a vLLM
mechanism is relevant but too broad for this target, name it as a future
target instead of implementing it here.

## Scope

In scope:

- refine the 07.65 owner split inside MoE/shared experts if necessary;
- audit whether `gate_up_proj` and `down_proj` are direct-copy heavy because
  of generic FP8 wrapper staging, activation quantization, dequantization,
  dtype casts, or layout materialization;
- test one or two narrow opt-in fixes against the current victory baseline;
- reuse the existing cached BF16 projection pattern for shared experts if the
  memory and correctness gates make sense;
- test whether runner finalization can avoid redundant `.float()` or `.to()`
  materialization without changing accepted numerics;
- record all new cache/workspace memory costs in bytes/rank, GiB/rank, and
  equivalent KV-cache tokens/pages at page size 256;
- add a benchmark/text-smoke variant if a new opt-in path is introduced;
- preserve decode CUDA graph replay and keep eager decode at `0`.

Out of scope:

- INT8 MoE, W8A8/W8A16, or INT8 Tensor Core routes;
- new Marlin backend work;
- changing routed expert backend semantics;
- broad communication/NCCL rewrites;
- promoting the 07.64 metadata deforestation helper;
- full FP8 KV cache or packed `fp8_ds_mla`;
- unified cache/workspace manager implementation;
- broad graph/layout cleanup outside the named MoE/shared owners.

If INT8 MoE looks attractive during review, write a separate future target such
as `TARGET 07.xx: DSV4 SM80 INT8 MoE Feasibility`, with independent accuracy
and performance gates.

## Candidate Fixes

Try candidates in this order.  Stop once one candidate meets the gates and a
fresh profile confirms the new bottleneck order.

### 1. Shared Expert Cached BF16 Weight Path

Hypothesis: `shared_experts.gate_up_proj` and `shared_experts.down_proj` still
pay generic FP8 projection staging during graph replay.  Attention owners
previously improved from cached BF16 dequantized weights, and the same helper
shape may work here.

Implementation shape:

- add explicit opt-in toggles, for example:

  ```text
  MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1
  ```

- prepare shared expert gate/up and down BF16 dequantized weights before CUDA
  graph capture;
- forbid implicit rebuild inside forward after graph capture, matching the
  existing attention projection cache discipline;
- add per-owner cache reports under `prepare_for_cuda_graph_capture`;
- add a variant such as:

  ```text
  dsv4_sm80_a100_victory_sharedbf16
  ```

  to both perf matrix and text smoke if this opt-in path is implemented;
- record memory for gate/up and down separately.

Abort or keep as non-promoted if memory cost is disproportionate to measured
gain.  Do not hide a large cache behind the victory bundle without the memory
ledger.

### 2. Runner Finalization Boundary Cleanup

Hypothesis: the runner currently materializes routed and shared outputs through
fp32 and then converts back to bf16.  Some materializations may be required for
numerics or all-reduce, but some may be redundant after the current backend
already produces `[tokens, hidden]` outputs.

Implementation shape:

- compare current:

  ```text
  routed_output.float()
  shared.float()
  y = y + shared
  all_reduce(y)
  y.to(flat.dtype)
  ```

  against narrower alternatives;
- keep the promoted fp32 finalization path as the default until accuracy is
  proven;
- any bf16 finalization or bf16 all-reduce variant must be opt-in and must
  record output deltas versus the promoted path;
- do not accept a faster path that changes simple text-smoke behavior or has
  unexplained numerical drift.

### 3. Shared Expert Overlap Audit

Hypothesis: vLLM may hide some shared expert cost with `SharedExperts` overlap
or modular-kernel ownership.

This target may inspect and measure whether mini's shared expert work can be
overlapped with route/routed expert work, but do not implement a multi-stream
runtime redesign unless the first two candidates fail and the profile shows
shared expert compute, rather than copy/layout staging, remains dominant.

If overlap is the only plausible next step, write a follow-up target instead
of building it here.

## Work Plan

1. Create artifacts:

   ```text
   performance_milestones/target07_moe_shared_expert_staging_cleanup/
     README.md
     raw/
     summaries/
     scripts/
   ```

2. Freeze the inherited baseline.

   Record:

   - current git state;
   - current `dsv4_sm80_a100_victory` macro from 07.63;
   - 07.65 direct-copy owner table;
   - active opt-ins in the victory bundle;
   - whether 07.64 metadata deforestation is disabled.

3. Run or reuse a focused baseline profile.

   Prefer reusing 07.65 if no code changed.  If code changed or the owner split
   needs more detail, capture a fresh 4096/128/batch4 TP8 nsys profile with
   `MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1`.

4. Review vLLM source parity.

   Produce a short table:

   | Boundary | mini current | vLLM current | Adapt decision |
   | --- | --- | --- | --- |
   | shared gate/up projection | ... | ... | ... |
   | shared down projection | ... | ... | ... |
   | shared/routed finalization | ... | ... | ... |
   | shared expert stream/order | ... | ... | ... |

5. Implement the smallest justified opt-in.

   Preferred first implementation: shared expert cached BF16 weight path, if
   shape/memory/correctness checks pass.  If source review proves this is not
   the direct-copy source, choose a runner-finalization cleanup instead.

6. Add benchmark entry points.

   If a new runtime opt-in is added, update both:

   ```text
   benchmark/offline/deepseek_v4_perf_matrix.py
   benchmark/offline/deepseek_v4_text_smoke.py
   ```

   The variant should compose on top of `dsv4_sm80_a100_victory` and should not
   silently enable INT8 or unrelated precision paths.

7. Validate correctness.

   Required:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_text_smoke.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory <new_variant> \
     --output performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/text_smoke.json
   ```

   If a finalization precision boundary changes, add a focused tensor-output
   comparison against the promoted path and record the error distribution.

8. Measure performance.

   Required short macro:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory <new_variant> \
     --scenarios decode_throughput_bs8 \
     --prompt-len 4096 \
     --decode-len 128 \
     --batch-size 4 \
     --repeats 3 \
     --warmup-repeats 1 \
     --page-size 256 \
     --num-pages 128 \
     --output-dir performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/macro_4096x128_bs4_np128 \
     --keep-going
   ```

   Required long macro if the short macro or owner profile improves:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory <new_variant> \
     --scenarios decode_throughput_bs8 \
     --prompt-len 4096 \
     --decode-len 1024 \
     --batch-size 4 \
     --repeats 3 \
     --warmup-repeats 1 \
     --page-size 256 \
     --num-pages 128 \
     --output-dir performance_milestones/target07_moe_shared_expert_staging_cleanup/raw/macro_4096x1024_bs4_np128 \
     --keep-going
   ```

9. Capture a final profile.

   Capture 4096/128/batch4 rank0 with the new variant and classify the same
   direct-copy owners.  The report must show whether:

   - MoE/shared expert staging fell;
   - total direct_copy fell;
   - graph replay remained active;
   - the next bottleneck changed.

10. Write final README.

    Include:

    - exact commands;
    - git state;
    - vLLM source comparison;
    - implemented toggle/variant;
    - memory ledger;
    - correctness results;
    - macro results;
    - final owner table;
    - promote / keep opt-in / reject decision;
    - next target recommendation.

## Gates

Correctness gate:

- text smoke passes for the new variant;
- graph replay remains active;
- eager decode remains `0`;
- any finalization precision change has an explicit output-delta report.

Owner gate:

- preferred: MoE/shared expert direct-copy group falls by at least `0.10s` in
  the 4096/128/batch4 rank0 profile; or
- acceptable: the largest single owner, `shared_experts.gate_up_proj`, falls by
  at least `25%` and no adjacent owner grows enough to cancel it.

Macro gate:

- 4096/128/batch4 should not regress by more than `1%`;
- 4096/1024/batch4 should improve by at least `1%` for promotion into the
  next best path;
- if owner timing improves but macro gain is below `1%`, keep the path opt-in
  and stop with a clear explanation.

Memory gate:

- record incremental bytes/rank and equivalent KV tokens/pages;
- if a cache costs more than `1.0 GiB/rank`, require a stronger justification
  than a noise-level macro gain;
- do not add a large cache to `dsv4_sm80_a100_victory` without an explicit
  memory tradeoff decision.

Scope gate:

- no INT8 MoE;
- no broad MoE backend rewrite;
- no unrelated attention, metadata, NCCL, sampler, or KV-cache changes;
- no default behavior changes unless correctness, owner, macro, and memory
  gates all pass.

## Stop Conditions

Stop and write the report when:

- one scoped opt-in implementation clears the gates;
- one scoped implementation improves owner timing but misses macro promotion;
- shared expert staging is no longer a top-two direct-copy contributor after a
  fresh profile;
- the next promising change is INT8 MoE, multi-stream runtime redesign, or a
  broad cache/workspace manager;
- two focused cuts each produce less than `1%` macro gain and less than
  `0.05s` owner reduction;
- correctness becomes unstable after one focused fix attempt.

Do not keep polishing small local copies after the main owner group has been
tested.  Select the next target from the final profile.

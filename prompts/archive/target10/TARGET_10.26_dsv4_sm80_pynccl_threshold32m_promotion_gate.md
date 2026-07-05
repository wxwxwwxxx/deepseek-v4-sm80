# TARGET 10.26: DSV4 SM80 PyNCCL Threshold32M Promotion Gate

## Status

Run after TARGET 10.25.

TARGET 10.25 showed that the simple global PyNCCL threshold candidate repeated
positively, while explicit per-owner/per-size routing did not beat it in the
cheap no-weight replay gate. This target is a short promotion gate for that
candidate.

Do not explore new communication backends here.

## Goal

Decide whether this opt-in should become the recommended or promoted
communication path for DeepSeek V4 Flash TP8/A100/sm80:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

The key question is:

```text
Is PyNCCL threshold32m repeat-stable across all required macro scenarios, and
does owner timing/profile confirm that the gain comes from communication
without breaking text correctness or CUDA graph replay?
```

## Required Inputs

Read first:

- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/archive/target10/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md`
- `prompts/archive/target10/TARGET_10.2_dsv4_sm80_comm_stack_backend_experiments.md`
- `prompts/archive/target10/TARGET_10.25_dsv4_sm80_comm_size_owner_routing.md`
- `performance_milestones/target10_moe_reduce_bf16_parity/README.md`
- `performance_milestones/target10_comm_stack_backend_experiments/README.md`
- `performance_milestones/target10_comm_size_owner_routing/README.md`

Mini references:

- `python/minisgl/distributed/impl.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`
- `python/minisgl/env.py`
- `python/minisgl/engine/engine.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

## Fixed Candidate

Baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
```

Candidate:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

Common runtime:

```text
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Keep Torch/NCCL as the fallback. Do not add a new route policy in this target.

## Work Plan

### 1. Sanity And Text Smoke

Run text smoke for the candidate:

- page size `256`;
- graph buckets `1 2 4 8 16`;
- prefix cache and component ownership enabled;
- fail on warnings if the smoke script supports it.

Record generated outputs and graph replay/eager.

### 2. Repeat-Stable Macro Gate

Run same-run A/B against the Torch/NCCL fixed-BF16 baseline.

Required scenarios:

- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- `serving_mixed_112req_wave16`;
- `prefix_multi_112req_wave16`.

Minimum repeats:

- at least `2` repeats for every scenario;
- use a third repeat for any scenario whose delta is near zero, noisy, or
  contradicts TARGET 10.25.

Record:

- elapsed;
- output tok/s;
- decode tok/s;
- graph replay/eager;
- communication stats;
- prefix hit rate for prefix scenarios;
- per-repeat deltas.

### 3. Owner Timing And Profile

Only if the repeat-stable macro gate is positive or borderline-positive, run
owner timing/profile on the most decision-relevant scenarios.

Minimum:

- one owner-timing run for `historical_4096_1024_bs4`;
- one owner-timing or profile run for `serving_mixed_112req_wave16`;
- include `historical_4096_128_bs4` if this scenario is noisy or regresses.

Profile goal:

- confirm that communication owner time improves or at least does not regress;
- confirm graph replay remains zero-eager;
- confirm NCCL kernel family/count is consistent with PyNCCL threshold32m;
- confirm there is no new large D2D copy amplification outside the known
  symmetric-buffer path.

Do not spend the target on deep Nsight archaeology if the macro gate is clearly
negative.

### 4. Promotion Decision

Classify the candidate as one of:

- `promote`: make it part of the recommended A100/sm80 DSV4 path;
- `recommended opt-in`: document it as the preferred high-performance
  communication opt-in but keep Torch/NCCL default;
- `keep experimental opt-in`: useful in some runs but not stable enough;
- `reject`: not repeat-stable or causes correctness/graph issues.

Promotion bar:

- text smoke passes;
- graph replay remains zero-eager;
- repeat-stable macro improvement is at least `2%` E2E on the main long decode
  or serving scenario, without a material regression on `4096x128` or prefix;
- owner timing/profile supports that the gain comes from communication or at
  least shows no hidden regression;
- fallback to Torch/NCCL remains simple.

If the candidate is promoted, update the relevant prompt docs to describe the
new recommended communication flags. If only recommended opt-in, do not change
the bundle defaults; record the command block clearly.

## Deliverables

Write:

```text
performance_milestones/target10_pynccl_threshold32m_promotion_gate/README.md
```

Include:

- candidate and baseline command/env blocks;
- text smoke result;
- repeat-stable macro table with per-repeat rows;
- communication stats before/after;
- graph replay/eager table;
- owner timing/profile summary;
- D2D copy accounting if profile data shows it;
- promote/recommended-opt-in/keep-experimental/reject decision;
- exact next-step recommendation.

## Done Criteria

Done when one of these is true:

- PyNCCL threshold32m is promoted or documented as the recommended opt-in;
- PyNCCL threshold32m is rejected or kept experimental due to repeat instability;
- macro evidence is positive but profile evidence is missing, and the report
  explicitly says what profile gate remains before promotion.

## Stop Rules

Stop and report instead of broadening if:

- text smoke fails;
- graph replay breaks;
- macro repeat gate is clearly negative;
- owner timing/profile contradicts the macro gain;
- fixing a failure requires changing PyNCCL internals, porting vLLM custom
  all-reduce, changing precision, or changing attention/prefix-cache code.

## Non-Goals

- New per-owner/per-size routing implementation.
- New vLLM custom all-reduce port.
- CUDA P2P/IPC custom collective work.
- Low-precision model changes.
- Attention kernel work.
- Prefix/SWA ownership work.

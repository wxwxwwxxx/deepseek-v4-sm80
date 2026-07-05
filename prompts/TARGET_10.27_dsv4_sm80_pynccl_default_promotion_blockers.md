# TARGET 10.27: DSV4 SM80 PyNCCL Default-Promotion Blockers

## Status

Run after TARGET 10.26.

TARGET 10.26 made the fixed communication candidate a **recommended opt-in**:

```text
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

The macro gate is repeat-stable and positive, but two evidence gaps block
default promotion:

1. `lm_head_all_gather` owner timing shows a very large non-captured total-time
   regression under PyNCCL, while captured hot-path time is neutral and macro
   throughput still wins.
2. The full serving Nsight attempt from TARGET 10.26 did not save CUDA activity;
   kernel evidence came from a clean representative PyNCCL probe, not a
   full-model serving trace.

This target exists to resolve those two blockers, not to search for a new
communication backend.

## Goal

Decide whether PyNCCL threshold32m can safely become the default A100/sm80 DSV4
communication optimization, and if yes, update the relevant defaults/docs.

The final decision must be one of:

- `default promote`: make the candidate part of the recommended default
  A100/sm80 DSV4 path or bundle;
- `recommended opt-in`: keep the TARGET 10.26 decision because one blocker is
  still unresolved but macro evidence remains positive;
- `do not promote`: keep Torch/NCCL default because the anomaly is a real
  regression or the full-model profile contradicts the macro wins.

## Required Inputs

Read first:

- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`
- `prompts/TARGET_10.26_dsv4_sm80_pynccl_threshold32m_promotion_gate.md`
- `performance_milestones/target10_pynccl_threshold32m_promotion_gate/README.md`

Mini references:

- `python/minisgl/distributed/impl.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`
- `python/minisgl/utils/dsv4_owner_timing.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

Useful historical profiling references:

- `performance_milestones/target07_vllm_gap/scripts/nsys_rank_wrapper.sh`
- `performance_milestones/target07_vllm_gap/scripts/nsys_mini_4096x128_bs4_fair.sh`
- `performance_milestones/target07_vllm_gap/scripts/summarize_nsys_sqlite.py`

## Fixed Candidate

Baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
# Torch/NCCL benchmark path; no --use-pynccl
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

Use the TARGET 10.26 macro numbers as the prior. Only rerun full macro if a
change is made or a blocker investigation finds a reason to invalidate those
numbers.

## Work Plan

### 1. Reproduce And Localize `lm_head_all_gather`

Re-run the smallest owner-timing A/B needed to reproduce the anomaly:

- `historical_4096_1024_bs4` is required;
- add `serving_mixed_112req_wave16` if the first run is ambiguous;
- keep the fixed BF16 MoE reduce baseline and the PyNCCL threshold32m
  candidate exactly as above.

Collect enough raw timing detail to answer:

- Is the large PyNCCL total-time increase entirely from non-captured CUDA event
  samples?
- Which shapes, sequence ranges, and phases contribute the extra time?
- Does `cuda_by_label_shape` show one pathological shape or a broad all-gather
  problem?
- Do raw `cuda_samples` for `dsv4.owner.comm.dsv4.lm_head_all_gather` contain
  a small number of outliers, capture-time samples, shutdown samples, or repeated
  graph-capture samples?
- Does the anomaly appear when owner timing is repeated in a fresh torchrun?

Suggested instrumentation settings:

```bash
MINISGL_DSV4_OWNER_TIMING=1
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000
```

`MINISGL_DSV4_OWNER_TIMING_SYNC_HOST=1` may be used as a secondary check, but
do not over-interpret it: the `lm_head_all_gather` table in TARGET 10.26 came
from CUDA event ranges, while sync-host primarily affects host ranges.

If the current summary format hides the needed evidence, add a narrow report
helper or a small owner-timing export change. Keep it focused on:

- captured versus non-captured split;
- label and shape split;
- first/last sample sequence windows;
- top outlier samples for a single label.

Do not rewrite owner timing broadly in this target.

### 2. Probe All-Gather Outside The Full Model

Build or reuse a small TP8 probe that exercises the same communication wrapper
as the model:

```text
DistributedCommunicator.all_gather(..., label="dsv4.lm_head_all_gather")
```

Test both backends:

- Torch/NCCL;
- PyNCCL threshold32m.

Use shapes and dtypes matching the real lm-head path where possible:

- FP32 logits from `DSV4VocabParallelEmbedding.linear`;
- local vocab shard shape from the model config;
- representative batch sizes from `historical_4096_1024_bs4` and
  `serving_mixed_112req_wave16`.

Run graph and non-graph variants if the probe can do so cheaply. The probe
should distinguish three outcomes:

- `measurement artifact`: standalone all-gather is neutral and only the model
  owner report is odd;
- `PyNCCL all_gather regression`: standalone all-gather reproduces the
  regression;
- `model phase attribution issue`: standalone all-gather is neutral, but model
  non-captured phases are expensive for a clear reason.

If a real PyNCCL `lm_head_all_gather` regression is proven, do the smallest
rollback-compatible fix:

- keep PyNCCL for the hot all-reduce owners;
- route `lm_head_all_gather` back to Torch/NCCL only if the macro gate confirms
  that this mixed path beats both pure Torch/NCCL and pure PyNCCL;
- document this as a follow-up or local fix, not a broad owner-routing system.

### 3. Repair Full-Model Nsight Capture

Create a target-local Nsight script under:

```text
performance_milestones/target10_pynccl_default_promotion_blockers/scripts/
```

Prefer the rank-wrapper pattern from TARGET 07:

```text
torchrun --standalone --nproc_per_node=8 --no-python \
  <nsys_rank_wrapper.sh> python benchmark/offline/deepseek_v4_perf_matrix.py ...
```

Profile rank `0` first; add rank `1` or all ranks only if needed. Use a short
but representative workload:

- first choice: `serving_mixed_112req_wave16` with reduced repeats if runtime is
  too long;
- fallback: `historical_4096_128_bs4` if it is the only workload that exits
  cleanly under Nsight.

Minimum Nsight settings:

```text
-t cuda,nvtx,osrt,cublas
--sample=none
--cpuctxsw=none
--backtrace=none
--cudabacktrace=none
--trace-fork-before-exec=true if this Nsight version supports it
```

After export, verify the sqlite contains CUDA activity, not only OS runtime:

- CUDA GPU kernel summary table is non-empty;
- CUDA memcpy summary table is present or explicitly empty;
- NCCL kernels appear for the communication owners;
- CUDA graph replay kernels or graph launches are visible enough to confirm the
  profiled path is the model path.

The profile summary must answer:

- Do small BF16 all-reduces use the expected PyNCCL symmetric path?
- Do large BF16 all-reduces use the expected direct NCCL path?
- Does `lm_head_all_gather` use direct all-gather output without unexpected D2D
  copy amplification?
- Is there any full-model CUDA activity that contradicts the TARGET 10.26 clean
  representative probe?

If a full serving profile still cannot be captured, a rank-scoped full-model
profile is acceptable only if it has CUDA activity and enough NVTX/owner context
to validate the candidate. Record the limitation clearly.

### 4. Promotion Or Rollback Decision

Default promotion is allowed only if all are true:

- TARGET 10.26 macro wins remain valid or are revalidated after any change;
- text smoke still passes;
- graph replay remains zero-eager;
- `lm_head_all_gather` anomaly is explained as harmless measurement/phase
  attribution, or fixed with a mixed path that passes macro and smoke gates;
- full-model or rank-scoped Nsight captures CUDA activity and supports the
  expected PyNCCL threshold behavior;
- rollback remains simple.

If default promotion is accepted, update the relevant prompt docs and, if the
codebase has a clear A100/sm80 bundle or benchmark preset for this path, update
that preset. Keep an explicit rollback note:

```bash
unset MINISGL_PYNCCL_MAX_BUFFER_SIZE
# remove --use-pynccl or use the Torch/NCCL preset
```

If default promotion is rejected, preserve the TARGET 10.26 recommended opt-in
command and explain exactly which blocker remains.

## Deliverables

Write:

```text
performance_milestones/target10_pynccl_default_promotion_blockers/README.md
```

Include:

- reproduction table for `lm_head_all_gather`;
- captured/non-captured split;
- shape and outlier breakdown;
- standalone all-gather probe result;
- full-model or rank-scoped Nsight artifact links;
- kernel and memcpy summary;
- macro recheck only if code or default behavior changed;
- final decision: `default promote`, `recommended opt-in`, or `do not promote`;
- exact next step.

Large `.nsys-rep`/sqlite files should be symlinked under the milestone `raw/`
directory. Small summaries and scripts can be copied directly.

## Done Criteria

Done when one of these is true:

- PyNCCL threshold32m is default-promoted with both blockers resolved;
- PyNCCL threshold32m remains recommended opt-in with a precise unresolved
  blocker;
- PyNCCL threshold32m is not promoted because a real all-gather/profile
  regression is proven.

## Stop Rules

Stop and report instead of broadening if:

- text smoke fails;
- graph replay breaks;
- the all-gather anomaly is real and fixing it requires a broad communication
  routing redesign;
- Nsight cannot produce CUDA activity after a rank-wrapper full-model attempt
  and one simpler representative full-model attempt;
- the target starts drifting into low precision, attention kernels, prefix/SWA
  ownership, or vLLM custom all-reduce porting.

## Non-Goals

- New communication backend exploration.
- Broad per-owner/per-size routing.
- vLLM custom all-reduce port.
- CUDA P2P/IPC collective development.
- Low-precision model changes.
- Attention kernel work.
- Prefix/SWA ownership work.

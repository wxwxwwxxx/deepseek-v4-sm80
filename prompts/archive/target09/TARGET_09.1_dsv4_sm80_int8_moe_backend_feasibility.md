# TARGET 09.1: DSV4 SM80 INT8 MoE Backend Feasibility

## Status

Ready after TARGET 08 small-kernel cleanup is stopped and TARGET 09 is resumed.

This target is **microbench and feasibility only**.  Do not replace the
production MoE path in this target, and do not run long E2E macro benchmarks
until the microbench gates below have passed.

## Goal

Determine whether an INT8 routed-MoE path can beat the current exact-ish
MXFP4/WNA16 Marlin MoE path on A100/sm80.

The question is not "can mini run an int8-looking path?"  The question is:

```text
Can a real SM80 INT8 Tensor Core grouped-MoE backend beat the promoted
WNA16/Marlin path after routing, activation quantization, W13, SiLU/mul,
W2, dequant/finalization, and memory traffic are counted?
```

The result should be a go/no-go decision for TARGET 09.2 opt-in integration.

## Baseline

Use the current promoted exact/capacity baseline as the performance reference:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect_replaymetafused
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL threshold32m default
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Mini uses system Python from `/workspace/mini-sglang`.  vLLM references, if
runtime checks are needed, use:

```text
/workspace/vllm-dsv4-docker
/workspace/venvs/vllm-dsv4/bin/python
```

## Baseline Facts

Mini's current DSV4 routed expert path is tensor-parallel over expert
intermediate dimension:

- each TP rank owns all experts but only local W13/W2 intermediate shards;
- routed experts produce a partial `[tokens, hidden]` output per rank;
- the promoted runner combines routed and shared experts locally;
- one final `dsv4.v1_moe_reduce_once_all_reduce` reduces the combined MoE
  output;
- TARGET 10 casts this final reduce to BF16 by default.

Therefore this target must keep INT8/INT32 internal to the candidate expert
backend and emit BF16 or FP32 at the TP reduce boundary.  INT8 communication is
TARGET 09.25 and is out of scope here.

## Idea Review

The INT8 MoE idea is plausible on A100 because SM80 has INT8 Tensor Cores while
it has no native FP4 Tensor Cores.  However, the current WNA16/Marlin path is a
strong baseline and already avoids the old FP4 reference bottleneck.  A useful
INT8 path must avoid these traps:

- it must not insert standalone quant/dequant kernels whose HBM traffic is
  larger than the GEMM win;
- it must not silently fall back to BF16/FP32 GEMM;
- it must support grouped/routed MoE, not only dense GEMM;
- it must handle both W13/up-gate and W2/down shapes;
- it must preserve route weights, top-k behavior, and TP local intermediate
  layout;
- it must leave a BF16/FP32 boundary before MoE all-reduce unless TARGET 09.25
  separately proves INT8 communication.

The user's FP4-in-kernel LUT idea is worth testing, but it is a research
candidate, not an assumption.  The target should evaluate two different ways to
feed an INT8 Tensor Core backend:

1. **Prepacked INT8 weights on load**: convert FP4/MXFP4 expert weights to a
   backend-specific INT8 packed layout during model load/preparation, then
   release original FP4 GPU expert weights if safe.
2. **FP4-to-INT8 inside kernel**: keep compact FP4/MXFP4 weight storage and use
   a small LUT or scale path inside the GEMM backend to materialize INT8 values
   close to the Tensor Core load path.

The second route may save persistent memory and HBM reads, but it only wins if
the LUT/scale conversion is fused into the backend without reducing Tensor Core
occupancy or adding a memory staging bottleneck.

## References

Mini:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/marlin_wna16.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/kernel/triton/fused_moe.py`
- `python/minisgl/kernel/csrc/`

SGLang:

- `/workspace/sglang-main/python/sglang/jit_kernel/moe_wna16_marlin.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/gemm/marlin_moe/`
- `/workspace/sglang-main/python/sglang/jit_kernel/per_token_group_quant_8bit.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/trtllm_lora_temp/`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/int8.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/online/int8.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/gptq_marlin.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/awq_marlin.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`

Important caution: source review has already shown that a Marlin MoE path can
be WNA16/weight-only without supporting W8A8 INT8.  The target must prove the
actual backend dtype and kernel path, not just the backend name.

## Candidate Backend Priority

Evaluate candidates in this order:

1. **Mature Marlin-family grouped MoE backend** if it truly supports the needed
   INT8 route on SM80.  This is preferred only if it supports grouped MoE,
   DSV4 shapes, and real INT8 Tensor Core execution.
2. **Small modification to the current WNA16/Marlin bridge** if the code shape
   is close enough to add a packed INT8 or FP4-LUT-to-INT8 path without a broad
   rewrite.
3. **vLLM/SGLang INT8 MoE backend port or ABI bridge** if a real backend exists
   but mini cannot import it directly.
4. **Standalone Triton/CUDA microkernel** only as a proof-of-physics or oracle.
   It is not an integration candidate unless it supports grouped MoE and both
   W13/W2 with route weights.

Reject immediately as a performance candidate:

- torch fallback int8 matmul;
- dense-only GEMM without grouped MoE routing;
- path that dequantizes FP4 weights to BF16 and then quantizes to INT8 at every
  decode step;
- path that reports INT8 tensors but launches BF16/FP32 GEMM;
- path that needs full MoE E2E integration before any speed signal can be
  measured.

## Required Work

### 1. Shape and layout census

Record real TP8 DSV4 routed expert shapes:

- hidden size;
- number of routed experts;
- top-k;
- local intermediate size;
- tokens per expert distribution for representative decode buckets;
- W13/up-gate and W2/down dimensions;
- current FP4/MXFP4 checkpoint packed layout;
- current WNA16/Marlin packed layout, workspace, and memory footprint;
- original FP4 GPU expert memory, Marlin packed memory, and release behavior.

Use source inspection first.  Load weights only if shape/runtime facts cannot be
obtained cheaply from existing reports or config.

### 2. Backend census and truth test

For every candidate, report:

- grouped MoE support versus dense GEMM only;
- SM80 support;
- W8A8, W8A16, W4A16, or weight-only semantics;
- expected activation dtype and weight dtype;
- scale granularity and zero-point behavior;
- supported top-k/routing/finalize behavior;
- expected weight layout and repack API;
- mini Torch ABI/build feasibility;
- whether original FP4 expert weights can be released after preparation.

Truth-test the candidate by inspecting kernel launch names, dtype checks, or
minimal compile/run artifacts.  If possible, use Nsight Compute/Nsight Systems,
cuobjdump, or source-level evidence to show that the path uses INT8 Tensor Core
instructions or a known INT8 Tensor Core backend.

### 3. Weight conversion route study

Compare at least these two routes:

#### A. Prepacked INT8 on load

- Convert FP4/MXFP4 weights to INT8 packed backend format once during prepare.
- Measure conversion cost and peak memory.
- Record steady-state memory if original FP4 GPU expert weights are released.
- Keep a rollback/oracle path using current WNA16.

#### B. In-kernel FP4/MXFP4 LUT to INT8

- Keep compact FP4/MXFP4 weight storage.
- Use a small LUT and scale path inside or immediately adjacent to the backend
  load path to produce INT8 fragments.
- Avoid writing a full INT8 weight tensor to HBM every run.
- Measure whether the LUT path preserves Tensor Core occupancy and improves HBM
  traffic enough to offset added instructions.

If route B cannot be fused into the backend and requires a separate full-weight
materialization kernel per run, reject it.

### 4. Activation quantization boundary study

Measure candidate activation quantization boundaries:

- after route/dispatch before W13;
- fused with route dispatch;
- inside W13 backend input loading;
- after SiLU/mul before W2;
- fused with SiLU/mul and W2 input preparation;
- inside W2 backend input loading.

For each boundary, report:

- scale granularity;
- quantization error;
- HBM reads/writes;
- kernel launches;
- graph-capture compatibility;
- whether the output can remain BF16/FP32 before TP all-reduce.

Standalone quant/dequant kernels are allowed as attribution tools.  They are not
acceptable production candidates unless the complete MoE boundary still wins.

### 5. Microbench ladder

Run the smallest benchmarks first and stop early when gates fail.

#### Ladder A: backend-only no-weight or synthetic weights

- Compile/import candidate backend.
- Run synthetic real-shape W13 and W2 GEMMs.
- Compare with current WNA16/Marlin backend at matching shapes.
- Confirm output dtype and instruction/backend path.

#### Ladder B: one-layer routed-MoE microbench

- Use real DSV4 route shapes and top-k metadata.
- Include dispatch/alignment, W13, SiLU/mul, W2, route-weight finalize, and
  BF16/FP32 output.
- Compare against current WNA16/Marlin routed expert microbench.
- Measure only representative bucket sizes first, for example bs `1,2,4,8,16`
  decode-like token counts and at least one prefill-like token count.

#### Ladder C: no-weight partial-model replay

Only if Ladder B passes:

- run a no-weight or partial-model graph replay that mimics per-layer MoE
  ownership without loading full model weights;
- prove graph capture/replay remains possible;
- estimate E2E ceiling from per-layer owner savings.

#### Ladder D: narrow macro smoke

Only if Ladder C predicts a meaningful E2E win:

- run one short TP8 macro smoke, preferably `historical_4096_128_bs4`;
- do not run the full macro matrix in this target unless the smoke confirms
  the microbench prediction.

## Metrics To Report

For every candidate that reaches Ladder B, report:

- W13 time;
- activation quant time;
- SiLU/mul time;
- W2 time;
- finalize/dequant time;
- total MoE boundary time;
- kernel launch count;
- HBM traffic estimate;
- packed weight memory;
- temporary workspace;
- peak memory during conversion;
- steady-state memory after possible original-weight release;
- error versus current WNA16/BF16 oracle:
  - max/mean absolute error;
  - relative error;
  - top-k/token stability if practical;
  - small generated-text smoke only if a runtime path is temporarily wired.

Also include a roofline-style interpretation:

- expected INT8 Tensor Core compute ceiling on A100;
- estimated arithmetic intensity for candidate W13/W2;
- whether the candidate is compute-bound, memory-bound, launch-bound, or
  quant/dequant-bound.

## Gates

Pass gates for recommending TARGET 09.2:

- at least one candidate is a real SM80 grouped-MoE INT8 Tensor Core path;
- it supports both W13 and W2 DSV4 shapes;
- it handles route/top-k/finalize semantics or has a clear minimal bridge;
- complete one-layer MoE boundary microbench beats current WNA16/Marlin by at
  least `15%` including quant/dequant/finalize overhead;
- estimated E2E ceiling is at least `2%` on the dominant serving scenarios;
- numerical error is measured and not obviously catastrophic;
- TP reduce boundary remains BF16/FP32;
- graph capture compatibility has a plausible path;
- memory ledger is explicit, including whether original FP4 GPU weights can be
  released.

Stop gates:

- no real SM80 grouped-MoE W8A8 backend exists;
- backend silently falls back to BF16/FP32;
- only dense GEMM is fast and routed MoE boundary is not;
- activation quant/dequant overhead erases the W13/W2 win;
- FP4-to-INT8 conversion requires repeated full-weight HBM materialization;
- in-kernel LUT path reduces occupancy enough to lose to WNA16;
- complete MoE boundary speedup is below `10%`;
- estimated E2E improvement is below `1%`;
- correctness drift is large or cannot be characterized;
- the target starts a broad E2E integration before microbench gates pass.

## Deliverables

Write results under:

```text
performance_milestones/target09_int8_moe_backend_feasibility/
```

Include:

- `README.md` with go/no-go for TARGET 09.2;
- backend support matrix;
- DSV4 shape/layout table;
- current WNA16/Marlin baseline timings;
- candidate kernel truth-test notes;
- prepacked INT8 versus in-kernel FP4-LUT-to-INT8 analysis;
- activation quantization boundary timing;
- microbench tables and raw JSON/CSV;
- numerical error/oracle summary;
- memory ledger and release plan;
- roofline/traffic estimate;
- exact commands, env vars, and any temporary patches or branch notes.

## Non-Goals

- Promoting INT8 MoE by default.
- Running the full macro matrix before microbench gates pass.
- Changing communication dtype.
- Implementing INT8 all-reduce.
- Replacing the current WNA16 path.
- Broadly rewriting MoE routing or scheduling without backend evidence.

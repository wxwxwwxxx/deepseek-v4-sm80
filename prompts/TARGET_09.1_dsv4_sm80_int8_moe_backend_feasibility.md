# TARGET 09.1: DSV4 SM80 INT8 MoE Backend Feasibility

## Status

Ready after TARGET 09.0 if MoE remains a material bottleneck.

This target is feasibility and microbench only.  Do not replace the production
MoE path in this target.

## Goal

Determine whether an INT8 W8A8 routed-MoE backend can beat the current exact
MXFP4/WNA16 Marlin MoE path on A100/sm80.

The winning candidate must be a real SM80 Tensor Core path, not a slow fallback
wrapped in an INT8 name.

## Baseline Facts

Mini's current routed MoE is tensor-parallel over expert intermediate dim:

- each TP rank owns all experts but only local W13/W2 intermediate shards;
- routed output is a partial `[tokens, hidden]` tensor per rank;
- promoted path combines routed and shared outputs locally, then does one
  `dsv4.v1_moe_reduce_once_all_reduce`;
- TARGET 10 casts this final reduce to BF16.

First INT8 MoE candidates must keep INT8/INT32 internal to the expert backend
and emit BF16/FP32 at the TP reduce boundary.  INT8 communication belongs to
TARGET 09.25.

## References

Mini:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/marlin_wna16.py`
- `python/minisgl/kernel/moe_impl.py`
- `python/minisgl/kernel/triton/fused_moe.py`
- `python/minisgl/kernel/csrc/`

SGLang:

- `/workspace/sglang-main/python/sglang/jit_kernel/per_token_group_quant_8bit.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/gemm/marlin_moe/`
- `/workspace/sglang-main/python/sglang/jit_kernel/moe_wna16_marlin.py`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`

## Required Work

1. Shape and layout census

   Record DSV4 routed expert shapes under TP8:

   - hidden size;
   - routed expert count;
   - top-k;
   - local intermediate size;
   - W13/W2 packed FP4/MXFP4 layout;
   - current Marlin WNA16 packed layout and memory.

2. Backend census

   Identify real SM80 W8A8 grouped-MoE candidates from SGLang/vLLM/mini.
   For each candidate, report:

   - whether it supports grouped MoE, not only dense GEMM;
   - supported activations and scale formats;
   - expected weight layout;
   - whether it can compile against mini's Torch ABI;
   - whether it handles DSV4 top-k routing and local expert layout.

3. Weight conversion feasibility

   Prototype or specify FP4/MXFP4 -> INT8 packed weight conversion at load time.
   Report:

   - conversion algorithm and scale granularity;
   - one-time load cost;
   - memory ledger for original FP4 weights, INT8 packed weights, and optional
     rollback copies;
   - whether original FP4 GPU expert weights can be released after preparation.

4. Activation quantization boundary

   Compare candidate boundaries:

   - after route/dispatch before W13;
   - fused with route dispatch;
   - after SiLU/mul before W2;
   - inside backend kernels.

   Measure scale computation and HBM traffic.  Reject standalone quant/dequant
   paths that dominate expected Tensor Core savings.

5. Microbench and oracle

   Run backend microbench at real DSV4 token/expert shapes.  Compare against:

   - current Marlin WNA16 routed MoE backend;
   - BF16 oracle or current exact path for numerical checks.

   Include top-k weighted route behavior if practical.  A pure dense GEMM
   benchmark is useful but not sufficient.

## Gates

Pass if:

- at least one SM80 backend is real and import/buildable;
- W13 and W2 microbench show enough speedup to survive quant/dequant overhead;
- numerical error is measured against the exact baseline;
- TP reduce boundary remains BF16/FP32;
- memory ledger is explicit.

Stop if:

- no backend supports grouped MoE W8A8 on SM80;
- candidate silently falls back to BF16/FP32;
- FP4 -> INT8 conversion requires repeated runtime round trips;
- activation quant/dequant overhead erases the win;
- quality error is large or unexplained;
- estimated E2E improvement is below `1%`.

## Deliverables

Write results under:

```text
performance_milestones/target09_int8_moe_backend_feasibility/
```

Include:

- `README.md` with go/no-go for TARGET 09.2;
- backend support matrix;
- shape/layout table;
- weight conversion and memory ledger;
- activation quantization timing;
- microbench results and raw data;
- correctness/oracle summary;
- exact patches or branch notes if a temporary backend bridge was built.


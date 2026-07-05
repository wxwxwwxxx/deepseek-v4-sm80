# TARGET 09.25: DSV4 SM80 INT8 Communication Boundary Feasibility

## Status

Optional low-precision communication research under TARGET 09.

Run this only after a fresh profile or TARGET 09.1/09.2 evidence shows TP
communication remains a material bottleneck.  The default TARGET 10
communication path is still:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL threshold32m default
```

## Goal

Decide whether INT8 communication can safely reduce TP communication cost for
DSV4 on A100/sm80.

This target is not allowed to promote INT8 communication by default.  It should
produce one of:

- a rejected design with clear scale/overflow/performance evidence;
- a microbench-proven opt-in candidate ready for a later E2E integration target;
- a recommendation to keep BF16/FP32 communication and focus elsewhere.

## Why This Is Separate From INT8 MoE

Current mini DSV4 MoE is tensor-parallel over expert intermediate dimension:

- routed experts produce partial `[tokens, hidden]` outputs on each TP rank;
- the promoted runner combines routed and shared experts locally;
- it then does one final `dsv4.v1_moe_reduce_once_all_reduce`;
- TARGET 10 casts that final MoE reduce input to BF16 before PyNCCL.

INT8 MoE compute can be correct while communication remains BF16.  INT8
communication changes a different numerical boundary: it quantizes partial
hidden outputs before cross-rank summation.

## Current Source Facts

Mini surfaces:

- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/distributed/impl.py`
- `python/minisgl/kernel/pynccl.py`
- `python/minisgl/kernel/csrc/src/pynccl.cu`
- `python/minisgl/kernel/csrc/include/minisgl/nccl227.h`

Known facts to verify in the target:

- NCCL headers expose `ncclInt8` and `ncclUint8`;
- mini's current PyNCCL dtype map exposes only FP16, BF16, and FP32;
- `DistributedCommunicator` has no per-owner dtype routing today;
- raw INT8 all-reduce has overflow and scale semantics that are not equivalent
  to BF16 all-reduce unless carefully designed.

Primary owners to study:

- `dsv4.v1_moe_reduce_once_all_reduce`;
- `dsv4.routed_expert_all_reduce` and `dsv4.shared_expert_all_reduce` for
  legacy comparison only;
- `dsv4.attn.wo_b.row_parallel_projection_all_reduce`;
- `dsv4.embedding_all_reduce`;
- `dsv4.lm_head_all_gather` as an all-gather contrast, not an all-reduce target.

## Candidate Designs

### A. Raw NCCL INT8 All-Reduce Probe

Add a local experimental PyNCCL dtype map for signed/unsigned int8 and run pure
communication microbenches.

This is a probe, not a correctness design.  It can only become a candidate if:

- quantized values have shared scale semantics across ranks;
- overflow is impossible or explicitly bounded;
- dequantized output matches BF16 reduce within an accepted error envelope;
- it beats BF16 PyNCCL after quantize/dequantize overhead.

### B. INT8 All-Gather Plus Fused Dequant-Sum

Communicate int8 partial outputs and scales, then locally dequantize and sum.

This avoids int8-sum overflow but may increase per-rank traffic versus ring
all-reduce on TP8.  It is useful only if latency, fusion, or graph behavior wins
despite the traffic math.

### C. Symmetric-Memory Custom Quantized Reduce

Use PyNCCL/NCCL symmetric memory or a mini-owned CUDA kernel to exchange int8
partials and accumulate to BF16/FP32 output.

This is the most plausible route if communication bytes matter but raw NCCL
int8 all-reduce is numerically wrong.  It is also the largest engineering
surface, so require a strong microbench win before E2E integration.

## Required Work

1. Source and owner census

   - Collect current communication stats with the promoted TARGET 10 baseline.
   - Record dtype, shape, count, bytes, and captured/eager owner timing.
   - Identify which owners are bandwidth-sensitive versus latency-sensitive.

2. Numerical boundary study

   - Capture representative BF16/FP32 partial outputs before each candidate
     all-reduce.
   - Compute per-tensor, per-token, and per-channel scale candidates.
   - Report quantization error versus BF16 reduce.
   - Report raw int8 sum overflow risk for TP8.
   - Reject designs that require unbounded int8 accumulation.

3. Microbench

   Compare at least:

   - current BF16 PyNCCL threshold32m;
   - torch/NCCL BF16 fallback;
   - raw NCCL INT8 all-reduce probe if dtype map is added;
   - INT8 all-gather plus local fused dequant-sum if implemented;
   - symmetric-memory custom reduce only if a minimal prototype is practical.

   Use both synthetic sizes and real owner shapes from `snapshot_communication_stats()`.

4. Cost model

   For each candidate, report:

   - communication bytes/rank;
   - quantize/dequantize HBM traffic;
   - scale tensor bytes;
   - extra workspace;
   - graph capture compatibility;
   - estimated E2E ceiling from owner-time savings.

5. Optional no-weight replay

   Run no-weight or owner-level replay only if microbench shows a meaningful
   win.  Do not run full model E2E in this target unless the target has already
   proved the numerical and microbench gates.

## Gates

Pass gates:

- dtype support is explicit, not a silent fallback;
- BF16 baseline and INT8 candidates are measured in the same process topology;
- quantization error and overflow risk are reported;
- candidate beats BF16 PyNCCL by at least `15%` on the targeted owner shape
  including quant/dequant overhead, or has a compelling latency win on small
  captured shapes;
- graph capture compatibility is checked for any candidate that reaches replay.

Stop gates:

- raw int8 all-reduce overflows or requires unacceptable scale loss;
- quantize/dequantize overhead dominates the bytes saved;
- all-gather traffic loses to BF16 all-reduce traffic on TP8;
- symmetric-memory prototype is slower than current PyNCCL threshold32m;
- owner-level estimated E2E gain is below `1%`.

## Deliverables

Write results under:

```text
performance_milestones/target09_int8_comm_boundary_feasibility/
```

Include:

- `README.md` with recommendation;
- current communication owner table;
- dtype support table;
- quantization error and overflow table;
- microbench tables and raw JSON/CSV;
- any temporary patches or branch notes if PyNCCL dtype support was modified;
- next-target recommendation.

## Non-Goals

- Default-promoting INT8 communication.
- Combining INT8 communication with INT8 MoE compute integration.
- Changing TARGET 10 BF16 PyNCCL behavior.
- Replacing PyNCCL with a broad new communication stack.
- Running large quality evaluation beyond what is needed to decide feasibility.

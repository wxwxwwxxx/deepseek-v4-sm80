# TARGET 07.73: DSV4 SM80 vLLM Quantized-Linear Backend Feasibility

Date: 2026-07-02

## Goal

Decide whether any standalone vLLM-aligned quantized-linear backend is worth
integrating into mini's current DeepSeek V4 A100 projection path.

This target is deliberately bounded:

- do not modify the promoted runtime path unless a standalone backend clears
  the gates;
- do not build another mini-owned direct FP8 projection wrapper like TARGET
  07.72 unless a roofline and microbench show it is plausible;
- treat vLLM's quant/dequant placement, weight packing, scale layout, and
  activation policy as the baseline contract;
- include quant/dequant, repacking, scale conversion, and workspace cost in
  every candidate timing;
- produce a clear next decision: integrate a backend, open a custom-kernel R&D
  target, or pivot away from quantized-linear projection work.

The main question is not "can low precision be faster in theory?"  It is:

```text
On A100/sm80, for mini's real DSV4 decode-small projection shapes, can an
available vLLM-style quantized-linear backend beat the promoted cached BF16
path by enough to justify runtime integration and precision risk?
```

## Current Baseline

Use the promoted milestone path as the baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Known promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

Inactive opt-ins that must not be used as baseline unless explicitly testing
an ablation:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

## Starting Evidence

TARGET 07.72 tested a mini-owned direct FP8/custom projection-cache boundary
against the promoted cached BF16 path.  The numerical error was small, but all
direct FP8 candidates were much slower.

Representative M=4 rows:

| Owner | Cached BF16 ms | Direct FP8 ms | Result |
| --- | ---: | ---: | --- |
| WQA/WKV/compress | `0.044003` | `0.344534` | fail |
| `q_wqb` | `0.037778` | `0.151194` | fail |
| `wo_b` local | `0.038232` | `0.150913` | fail |
| shared expert gate/up | `0.043902` | `0.342883` | fail |
| shared expert down | `0.038083` | `0.102496` | fail |
| `wo_a` | `0.062519` | `0.342240` | fail |

Interpretation:

- a naive software-FP8 projection wrapper is not enough on SM80;
- cached BF16 already removed mini's old per-decode dequantized-weight
  bottleneck;
- the next useful test is not another local direct-FP8 wrapper, but a
  standalone feasibility pass over the actual vLLM quantized-linear backend
  family.

## vLLM Source References

Use `/workspace/vllm-dsv4-docker` as the primary reference.

Start with:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/linear.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fbgemm_fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/fp_quant.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/input_quant_fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/utils/w8a8_utils.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/kernels/linear/scaled_mm/
/workspace/vllm-dsv4-docker/vllm/model_executor/kernels/linear/mxfp8/
/workspace/vllm-dsv4-docker/vllm/model_executor/kernels/linear/mixed_precision/
```

Important source facts to verify and record:

- `Fp8LinearMethod` selects `MarlinFP8ScaledMMLinearKernel` on GPUs without
  native FP8 hardware when supported.
- `MarlinFP8ScaledMMLinearKernel` is a weight-only FP8 path for SM80-like
  devices; it repacks weights/scales and calls `marlin_gemm`.
- vLLM's Marlin FP8 helper explicitly rejects W8A8 activation quantization:
  `Marlin W8A8 is not supported.`
- `FBGEMMFp8LinearMethod` may route through Marlin, and its Marlin path does
  not quantize activations.
- `torch._scaled_mm` / W8A8-style paths may require per-tensor activation
  scales and may be hardware/backend limited on SM80.
- DeepSeek V4 uses additional custom FP8 paths for indexer/cache/attention,
  but full `fp8_ds_mla` KV-cache E2E remains out of scope here.

Do not assume these facts are sufficient.  The target must prove whether a
candidate backend can be imported, prepared, and timed on the actual A100
environment.

## Mini Source Areas

Read before writing any script:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/layers/linear.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
performance_milestones/target07_vllm_aligned_fp8_custom_projection_cache_boundary/README.md
performance_milestones/target07_projection_gemm_backend_owner_reattribution/README.md
performance_milestones/target07_bf16_small_gemm_backend_cluster/README.md
performance_milestones/target07_precision_boundary_pivot/README.md
```

## Candidate Backend Lanes

Try these as standalone backend experiments only.  Do not wire a candidate into
the runtime until it clears the standalone gates.

### Lane A: vLLM FP8 Marlin Weight-Only Linear

Hypothesis: vLLM's Marlin FP8 path may beat cached BF16 for decode-small real
DSV4 projection shapes because it uses compact FP8 weights with Marlin's
weight-only kernel, while avoiding slow activation quantization.

Required work:

- reproduce vLLM's weight/scale preparation and Marlin repacking as closely as
  possible;
- use vLLM's scale expansion and exponent-bias handling instead of inventing a
  new scale path;
- include one-time preparation time and persistent workspace bytes in the
  report, but compare decode steady-state latency without repeated repacking;
- confirm there is no per-token or per-replay allocation/repacking.

Representative sources:

```text
vllm/model_executor/layers/quantization/fp8.py
vllm/model_executor/kernels/linear/scaled_mm/marlin.py
vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py
```

### Lane B: vLLM FBGEMM FP8 / Marlin-Compatible Route

Hypothesis: vLLM's FBGEMM FP8 method may expose a slightly different
weight-scale contract or backend selection that is easier to adapt than the
plain `Fp8LinearMethod`.

Required work:

- check whether the backend actually runs on A100 in this environment;
- record whether activations are quantized or kept BF16;
- compare against Lane A on identical shapes and inputs.

Representative source:

```text
vllm/model_executor/layers/quantization/fbgemm_fp8.py
```

### Lane C: `torch._scaled_mm` / W8A8-Style Scaled MM

Hypothesis: a scaled-mm route may help if it can use a real fast backend on
SM80 and if its quant/dequant boundary is not dominated by activation
quantization.

Required work:

- first prove backend availability and dtype support on A100;
- measure activation quantization separately and included;
- reject the lane if it needs a slow standalone activation quant/dequant step
  that vLLM does not use for the corresponding SM80 path;
- record whether the result is a true vLLM-aligned route or only a diagnostic.

### Lane D: INT8 Tensor-Core Projection Probe

Hypothesis: INT8 tensor cores may offer a future projection route if quality
loss is acceptable and if quantization can be placed cheaply.

This is exploratory and must stay opt-in/standalone:

- look for a vLLM-aligned generic W8A8 or int8 projection backend first;
- if vLLM has only MoE-specific int8 paths, state that clearly and do not
  pretend it is an existing DeepSeek V4 projection backend;
- test only persistent weight quantization and clearly measured activation
  quantization;
- compare outputs against promoted cached BF16 and report error distribution;
- do not integrate this lane in 07.73 even if the microbench is promising.

If INT8 looks promising, the next step should be a separate precision target
with quality gates, not a silent promotion.

### Lane E: Custom Mini Kernel R&D Candidate

Only enter this lane if the vLLM baseline backend is unavailable or too slow
and the roofline analysis shows real headroom.

Examples of plausible custom surfaces:

- fused BF16 activation load + FP8 weight dequant + small GEMM without
  materializing BF16 weights;
- per-owner persistent scale cache with dequant fused into the matmul mainloop;
- specialized M=`1,4,8,16` projection kernels for the dominant shape cluster.

Do not implement a custom kernel in this target unless it is tiny and the
roofline says the current backend is far from the attainable bound.  Otherwise
write the custom plan as the next target.

## Shapes And Owners To Test

Use real DeepSeek V4 layer weights/scales from `/models/DeepSeek-V4-Flash`.
Start with one representative layer that exists in all tested owners, then
optionally repeat on another layer if results are surprising.

Minimum owner set:

| Owner | Why |
| --- | --- |
| WQA/WKV/compress | representative attention projection and vLLM fused boundary |
| `q_wqb` | promoted cached BF16 projection owner with stable prior microbench |
| `wo_b` local | row-parallel projection compute, measured without all-reduce |
| shared expert gate/up | shared expert projection owner after TARGET 07.66 |
| shared expert down | complementary shared expert owner |
| `wo_a` grouped projection | tests whether the `wo_a` boundary has a low-precision alternative |

Minimum token rows:

```text
M = 1, 4, 8, 16
```

Every timing table must include:

- promoted cached BF16 baseline;
- any vLLM-aligned backend candidate;
- dequant-on-the-fly diagnostic only if useful to prove a bad boundary;
- quant/dequant or activation quant cost, both separated and included;
- one-time preparation/repack time and persistent workspace/cache bytes.

## Quant / Dequant Policy

This target must be strict about boundary fairness.

Rules:

- If vLLM uses weight-only Marlin on SM80, the candidate should also keep
  activations in BF16 and not insert an activation FP8 quant step.
- If vLLM expands tensor-wise scales to channel-wise scales during
  `process_weights_after_loading`, do the same for the candidate.
- If vLLM fuses exponent bias into scales, do the same rather than using
  `exp`, `pow`, or a per-token scalar conversion in the replay path.
- If a conversion is only needed once at model load, account for it as load
  time and persistent memory, not per-token decode latency.
- If a conversion happens every decode replay, it must be included in the
  candidate latency and treated as a first-class bottleneck.
- Do not call a backend "faster" if it wins only after excluding the
  quant/dequant step required to feed it.

## Roofline Analysis Requirement

For any backend that is close but not clearly winning, and for any proposed
custom kernel, compute a simple A100 roofline estimate.

Record the assumed A100 constants.  Suggested conservative defaults:

```text
BF16 tensor core peak: 312 TFLOP/s
INT8 tensor core peak: 624 TOPS
FP32 peak: 19.5 TFLOP/s
TF32 tensor core peak: 156 TFLOP/s
HBM bandwidth: 1.55 TB/s
```

For each owner/shape:

```text
FLOPs ~= 2 * M * N * K
Bytes ~= read(X) + read(W or packed W) + read(scales) + write(Y) + workspace traffic
Arithmetic intensity = FLOPs / Bytes
Roofline time lower bound = max(FLOPs / peak_compute, Bytes / HBM_bandwidth)
Efficiency = roofline_lower_bound / measured_time
```

The report should say whether the candidate is:

- compute-bound enough that INT8/Marlin could plausibly help;
- bandwidth-bound on weights/scales, where compact weights may help;
- launch/overhead-bound for tiny M, where custom fusion or graph grouping may
  matter more than raw tensor-core throughput;
- already close enough to the roofline that further projection work is
  unlikely to be the best target.

## Work Plan

1. Create the milestone folder:

```text
performance_milestones/target07_vllm_quantized_linear_backend_feasibility/
```

Use subfolders:

```text
scripts/
raw/
summaries/
```

2. Write a source-parity summary:

```text
summaries/vllm_quantized_linear_backend_parity.md
```

It must table each backend candidate, its source file, expected quant/dequant
strategy, activation policy, weight/scale layout, A100 support, availability
in this environment, and whether it is a real DeepSeek V4 projection candidate
or only a diagnostic.

3. Write an availability smoke:

```text
scripts/check_quantized_linear_backend_availability.py
raw/backend_availability.json
```

The smoke should import or directly probe candidate ops in the current mini
environment.  If a candidate requires the vLLM virtualenv, record the command
for:

```text
/workspace/venvs/vllm-dsv4/bin/python
```

If a package or op is missing, do not silently skip it.  Record the missing
module/op and whether the user could reasonably install it.

4. Write a focused standalone microbench:

```text
scripts/focused_quantized_linear_backend_microbench.py
raw/focused_quantized_linear_backend_microbench.json
raw/focused_quantized_linear_backend_microbench.md
```

Requirements:

- load real model weights/scales;
- use deterministic random or captured real hidden inputs;
- measure M=`1,4,8,16`;
- compare against promoted cached BF16 baseline;
- include quant/dequant cost according to the vLLM-aligned boundary;
- report quality/error against cached BF16 outputs.

5. Write a roofline table:

```text
summaries/quantized_linear_roofline.md
```

Include the formula, constants, owner shapes, measured latency, theoretical
bound, efficiency, and interpretation.

6. Stop before runtime integration unless the gates pass.

If a backend passes, write the next integration plan and suggested toggle.
If no backend passes but roofline shows headroom, write the next custom-kernel
R&D plan.  If no backend passes and roofline does not support custom work,
recommend pivoting away from quantized-linear projection.

## Gates

Standalone backend gate:

- candidate runs on A100/sm80 without unsupported dtype/runtime errors;
- candidate covers at least two representative owners;
- candidate covers M=`1,4,8,16`;
- no per-decode allocation or repacking is required;
- quant/dequant cost is included according to the vLLM-aligned boundary;
- quality is acceptable against promoted cached BF16 outputs.

Performance gate:

- at least `>=15%` latency reduction over promoted cached BF16 on at least two
  representative owners at M=`1,4,8,16`; and
- no owner regresses by more than `10%` unless the report explains why that
  owner would not be integrated.

Custom-kernel gate:

- roofline indicates at least `1.5x` plausible latency headroom on a meaningful
  owner cluster; and
- the custom surface is large enough to plausibly move 4096/1024 macro by at
  least `3%`; and
- the proposed quant/dequant boundary does not introduce a new replay-time
  memory traffic bottleneck.

Runtime integration gate:

- do not integrate in this target by default;
- only produce a runtime implementation plan if a standalone backend passes
  the above gates.

## Required Final README

Write:

```text
performance_milestones/target07_vllm_quantized_linear_backend_feasibility/README.md
```

It must include:

- baseline variant and inactive opt-ins;
- vLLM source-parity table;
- availability/import table;
- tested owner/shape table;
- per-backend latency table with quant/dequant included;
- quality/error table;
- one-time preparation time and persistent memory/workspace ledger;
- A100 roofline table and interpretation;
- decision: integrate backend, open custom-kernel R&D, or pivot;
- exact next target recommendation;
- do-not-continue condition.

## Stop Conditions

Stop immediately and write the README if:

- no vLLM-aligned backend can run on A100 in the available environment;
- the only available backend requires a replay-time quant/dequant step that is
  slower than promoted cached BF16;
- no candidate beats cached BF16 by `>=15%` on at least two representative
  owners;
- roofline says the remaining headroom is too small for a custom kernel to
  matter;
- the next idea is full FP8 KV-cache E2E, HC/router precision, or INT8 MoE.

Those belong to separate targets with separate quality gates.

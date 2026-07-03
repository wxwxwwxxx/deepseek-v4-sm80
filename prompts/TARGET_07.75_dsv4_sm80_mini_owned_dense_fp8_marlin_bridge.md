# TARGET 07.75: DSV4 SM80 Mini-Owned Dense FP8 Marlin Extension Bridge

Date: 2026-07-03

## Goal

Build and validate a mini-owned dense FP8 Marlin W8A16 custom-op bridge for
DeepSeek V4 Flash on A100/sm80.

This target is intentionally narrower than TARGET 07.74.  Do not integrate the
op into the TP8 model runtime yet.  First prove that the dense FP8 Marlin op
surface can be built against mini's default torch ABI and can reproduce the
vLLM backend behavior on the Phase A dense owner subset.

Target runtime environment:

```text
/workspace/mini-sglang default Python
torch 2.9.1+cu128
CUDA 12.8
A100 sm80
```

The bridge must not depend on importing vLLM at runtime.

## Why This Target Exists

TARGET 07.73 found a positive standalone backend:

```text
vLLM FP8 Marlin W8A16 block linear
```

It passed the standalone gate on:

- `attn.q_wqb`;
- `attn.wo_b` local projection;
- shared experts down.

TARGET 07.74 then wired a default-off runtime opt-in through vLLM Python
helpers:

```text
MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1
dsv4_sm80_a100_victory_fp8marlinproj
```

Focused calls still looked strong, but TP8 runtime validation stopped at a
bridge/ABI gate:

- default mini Python has no vLLM package, and importing vLLM source hits
  `vllm._C` torch ABI mismatch;
- the vLLM venv can call vLLM Marlin helpers, but mixes torch
  `2.11.0+cu128` with global mini packages and exposed `sgl_kernel` and
  mini Marlin WNA16 ABI problems;
- TP8 smoke/profile/macro therefore did not clear the 07.74 gates.

The correct next move is not to keep debugging mixed environments.  Port the
small dense FP8 Marlin op surface that mini actually needs and build it against
mini's own torch.

## Preflight State

The environment check before this target found:

- default mini Python: `/usr/bin/python`;
- default torch: `2.9.1+cu128`;
- CUDA: `12.8`;
- `_GLIBCXX_USE_CXX11_ABI=True`;
- GPU capability: `sm80`;
- `nvcc`: `/usr/local/cuda/bin/nvcc`, CUDA compilation tools `12.8`;
- `gcc/g++`: `13.3.0`;
- `ninja`: available;
- `cmake`: not installed.

This is acceptable.  Follow mini's existing `torch.utils.cpp_extension.load`
pattern; do not require CMake.

Tiny default-env compile probes passed for both C++ and CUDA extensions.  The
full existing `marlin_wna16` cache may rebuild slowly and should not be reused
as the dense bridge cache.

## Source Map

vLLM dense Marlin source root:

```text
/workspace/vllm-dsv4-docker/csrc/quantization/marlin
```

Relevant dense files:

```text
marlin.cu
gptq_marlin_repack.cu
kernel.h
kernel_selector.h
marlin_template.h
marlin.cuh
marlin_dtypes.cuh
marlin_mma.h
dequant.h
sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu
```

Core binding helpers:

```text
/workspace/vllm-dsv4-docker/csrc/core/registration.h
/workspace/vllm-dsv4-docker/csrc/core/scalar_type.hpp
/workspace/vllm-dsv4-docker/csrc/torch_bindings.cpp
```

The dense schemas needed from vLLM are:

```text
marlin_gemm(
  Tensor a,
  Tensor? c_or_none,
  Tensor b_q_weight,
  Tensor? b_bias_or_none,
  Tensor b_scales,
  Tensor? a_scales,
  Tensor? global_scale,
  Tensor? b_zeros_or_none,
  Tensor? g_idx_or_none,
  Tensor? perm_or_none,
  Tensor workspace,
  int b_type_id,
  SymInt size_m,
  SymInt size_n,
  SymInt size_k,
  bool is_k_full,
  bool use_atomic_add,
  bool use_fp32_reduce,
  bool is_zp_float
) -> Tensor

gptq_marlin_repack(
  Tensor b_q_weight,
  Tensor perm,
  SymInt size_k,
  SymInt size_n,
  int num_bits,
  bool is_a_8bit
) -> Tensor
```

Existing mini vendor path:

```text
python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16
```

That path already contains some shared Marlin headers and WNA16 MoE code, but
it does not contain the dense `marlin.cu`, `kernel.h`, `kernel_selector.h`,
`marlin_template.h`, or the dense sm80 FP8 template kernel.  Add a new dense
vendor subset or extend the vendor tree cleanly; do not overload the existing
WNA16 extension name/cache.

## Non-Goals

Do not do these in this target:

- TP8 runtime integration;
- `dsv4_sm80_a100_victory` bundle promotion;
- broad FP8 KV cache or `fp8_ds_mla` E2E;
- `wo_a` grouped two-launch Marlin route;
- FBGEMM-derived per-channel conversion route;
- `torch._scaled_mm` FP8 on A100;
- INT8 W8A8 projection;
- INT8 W8A8 MoE experiments or quality/performance gates;
- routed MoE backend changes;
- debugging vLLM venv as the primary route.

## Future INT8 W8A8 And TVM FFI Notes

There is a plausible future MoE INT8 W8A8 route, but it should not be bundled
into this dense FP8 Marlin bridge gate.

Reasons:

- dense FP8 Marlin and MoE INT8 W8A8 have different precision contracts,
  kernels, quality risks, and benchmark gates;
- TARGET 07.73 already rejected INT8 W8A8 as a first dense projection route;
- vLLM's Marlin MoE path supports int4/int8 weight-only style kernels, but its
  W8A8-INT8 path is not simply "turn on Marlin"; vLLM explicitly routes online
  expert int8 through its fused-MoE/oracle machinery;
- compiling broad extra kernels now can turn a focused ABI bridge into a
  source-porting target with unclear stop conditions.

Useful future source references:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/experts_int8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/online/int8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/int8.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/config.py
/workspace/vllm-dsv4-docker/csrc/quantization/w8a8/int8/scaled_quant.cu
/workspace/vllm-dsv4-docker/csrc/moe/marlin_moe_wna16/
```

07.75 may include a short compile inventory note for future INT8 W8A8 work,
but it must not compile those kernels into the dense FP8 bridge unless doing so
is strictly header-only/shared-infrastructure and cannot affect build time,
symbol registration, op behavior, or success gates.  If a separate source file,
schema, or runtime wrapper is needed, leave it for a later INT8 MoE target.

There is also future value in a unified TVM FFI/custom-op bridge target.  Mini
already uses `tvm_ffi` for several local kernels, while this target deliberately
uses `torch.utils.cpp_extension.load` to match existing Marlin WNA16 practice
and minimize ABI risk.  A later target can evaluate whether to expose compiled
Marlin/MoE ops through TVM FFI or a common op registry.

Expected TVM FFI benefits are mostly engineering benefits:

- one loader/registry surface for mini-owned C++/CUDA kernels;
- cleaner workspace/cache ownership and preallocation;
- fewer ad hoc Python extension modules;
- easier op availability probes and artifact reporting.

Do not assume TVM FFI by itself improves GPU kernel runtime.  Any performance
gain would come indirectly from cleaner capture boundaries, fewer Python-side
dispatch decisions, and better workspace lifetime management.  Measure this in
a dedicated future target after the dense FP8 bridge and any INT8 MoE
feasibility work have a stable op surface.

## Implementation Plan

Create artifacts:

```text
performance_milestones/target07_mini_owned_dense_fp8_marlin_bridge/
  README.md
  raw/
  scripts/
  summaries/
```

### 1. Add A Mini-Owned Extension Module

Suggested module:

```text
python/minisgl/kernel/dense_fp8_marlin.py
```

Suggested extension name:

```text
minisgl_dense_fp8_marlin
```

Suggested build directory:

```text
~/.cache/minisgl/dense_fp8_marlin
```

Do not reuse:

```text
minisgl_marlin_wna16
~/.cache/minisgl/marlin_wna16
```

Use `torch.utils.cpp_extension.load`, similar to
`python/minisgl/kernel/marlin_wna16.py`.

Required CUDA flags:

```text
-O3
-std=c++17
--expt-relaxed-constexpr
-static-global-template-stub=false
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_80,code=compute_80
```

### 2. Vendor Only The Required Dense Source Surface

Copy or adapt the minimum Apache-2.0 vLLM source subset needed to build dense
FP8 Marlin for BF16 activation and `fe4m3fn` FP8 weights on sm80.

Keep source provenance clear in comments or README notes.  Avoid pulling
unrelated vLLM kernels into mini.

The first build target should register only:

- `gptq_marlin_repack`;
- `marlin_gemm`.

If vLLM's binding file is too broad, create a small mini binding file that
registers these two schemas and calls the dense Marlin implementations.

### 3. Port The Minimal Python Packing/Apply Helpers

Implement mini-local helpers equivalent to the parts of vLLM used in 07.73:

- native DeepSeek block-FP8 weight handling;
- `pack_fp8_to_int32` style packing;
- block scale processing/permutation;
- FP8 exponent-bias scale fusion when required by the Marlin contract;
- workspace allocation;
- `prepare_dense_fp8_marlin_weight(...)`;
- `apply_dense_fp8_marlin_linear(...)`.

Keep activations BF16.  Do not insert runtime activation FP8 quantization.

The helper should be usable from focused scripts without importing vLLM.

### 4. Focused Correctness And Speed Probes

Run focused probes using real DSV4 tensors for:

- `attn.q_wqb`;
- `attn.wo_b` local projection;
- shared experts down.

Shapes:

```text
M = 1, 4, 8, 16
```

Compare three paths when possible:

1. promoted cached BF16 `F.linear` baseline;
2. mini-owned dense FP8 Marlin bridge;
3. vLLM helper path from `/workspace/venvs/vllm-dsv4` as an offline reference,
   not as a runtime dependency.

Record:

- latency median/p20/p80 or equivalent stable summary;
- max abs, mean abs, p99 abs;
- cosine similarity;
- dtype, shape, owner name, source weight path;
- one-time prepare/repack latency separately from steady-state GEMM latency.

Expected rough performance anchor:

- 07.73/07.74 Marlin steady-state absolute time was around `0.064 ms` for the
  strongest M=`4` dense owners.

Do not overfit to a single number.  The gate is that mini-owned Marlin should
be close to the vLLM helper path and materially faster than the same-run cached
BF16 baseline for the selected owners.

### 5. Memory And Build Ledger

The README must report:

- compiled extension path;
- source files used;
- torch version and ABI;
- build flags;
- first-build time;
- steady import/load time if measurable;
- persistent packed weight/scales bytes for each tested owner;
- workspace bytes;
- whether original FP8 tensors are needed after packing.

Use this target to prepare the memory logic for the later runtime integration,
but do not modify the TP8 model lifecycle unless needed for focused probes.

## Success Gates

This target succeeds only if all are true:

- default mini Python can build and import `minisgl_dense_fp8_marlin`;
- the extension does not import vLLM and does not depend on `sgl_kernel`;
- `gptq_marlin_repack` and `marlin_gemm` can run on A100/sm80;
- focused real-weight probes pass quality checks for the Phase A owners;
- steady-state latency is close to the vLLM helper path and faster than the
  same-run cached BF16 baseline on the selected owners;
- README records exact source files, build flags, ABI, performance, quality,
  and memory results.

If the op builds but is slower than vLLM's helper path, stop and explain the
delta before touching model runtime.  If the op cannot build, stop with the
first missing source/header/symbol and the smallest next fix.

## Stop Rules

Stop after the op bridge and focused probes pass.  Do not keep going into TP8
runtime integration inside this thread.

Stop early if any of these happen:

- compile failure shows a missing source/header dependency that needs a source
  import decision;
- link failure shows ABI mismatch against mini torch;
- op output diverges from the vLLM helper/reference beyond acceptable numerical
  noise;
- focused latency is not competitive with the vLLM helper path.

The next target after a successful 07.75 should revise TARGET 07.74's runtime
opt-in to use the mini-owned bridge instead of the vLLM Python bridge, then run
TP8 smoke, graph replay, profile, macro, and memory-lifecycle gates.

## Suggested Final README Outline

```text
# TARGET 07.75: Mini-Owned Dense FP8 Marlin Extension Bridge

Status:

## Environment
## Source Surface And Build
## Registered Ops
## Focused Owner Quality
## Focused Owner Latency
## Memory / Workspace Ledger
## Comparison Against vLLM Helper
## Decision
## Next Target
```

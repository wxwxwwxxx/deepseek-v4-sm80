# vLLM Marlin WNA16 Source Subset

This directory vendors the minimum source subset used by
`TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port` to build a mini-owned
Marlin WNA16 MoE custom-op probe.

Source origin:

- Upstream checkout used for this target:
  `/workspace/vllm-dsv4-docker`
- Components:
  - `csrc/core/registration.h`
  - `csrc/core/scalar_type.hpp`
  - `csrc/quantization/marlin/{dequant.h,gptq_marlin_repack.cu,marlin.cuh,marlin_dtypes.cuh,marlin_mma.h}`
  - `csrc/moe/marlin_moe_wna16/{kernel.h,kernel_selector.h,marlin_template.h,ops.cu,sm80_kernel_*.cu}`

The copied vLLM/Marlin source is Apache-2.0 licensed. Preserve upstream
copyright/license headers in source files when modifying this subtree.

This is intentionally narrower than vLLM's full CUDA extension. It is not a
general Marlin backend package; it exists to prove and then integrate the DSV4
SM80 MXFP4 W4A16 MoE path without a vLLM runtime dependency.

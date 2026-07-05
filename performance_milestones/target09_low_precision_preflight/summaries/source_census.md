### INT8 MoE

- Status: partial reference path; not ready as next integration target
- Evidence: source-derived
- Finding: vLLM has online per-row INT8 MoE loading and a Triton INT8 MoE backend, including W8A16/W8A8 config constructors. The Marlin path is only WNA16-like for this repo and vLLM explicitly asserts that W8A8 INT8 is not supported by Marlin. Mini's current Marlin wrapper requires fp16/bf16 activations.
- Sources:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/online/int8.py:30`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/int8.py:32`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/gptq_marlin.py:535`
  - `python/minisgl/kernel/marlin_wna16.py:250`

### FP8 KV/cache

- Status: strong DSv4-specific reference path
- Evidence: source-derived plus runtime memory pressure
- Finding: vLLM DeepSeek V4 only accepts fp8/fp8_ds_mla KV cache for its sparse FlashMLA path, with paged uint8 cache specs, FP8 quantize/insert, gather/dequant, and an SM80 reference fallback. SGLang has DSv4 MLA FP8 pack/quant/store kernels. This is the most mature low-precision route to study next, but it still needs a parity ledger before implementation.
- Sources:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1144`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1189`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py:7`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py:160`
  - `/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py:1`
  - `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py:12`

### INT8 communication

- Status: not ready
- Evidence: runtime-proven BF16 traffic plus source-derived lack of CUDA DSv4 path
- Finding: Runtime communication entries are BF16 for hidden all-reduces and FP32 for lm_head all-gather. Mini PyNCCL maps only fp16/bf16/fp32. SGLang exposes quant_all_reduce, but the source marks it as NPU support only and falls back to normal all-reduce on other devices. No DSv4 SM80 CUDA INT8 communication protocol was found.
- Sources:
  - `python/minisgl/kernel/csrc/src/pynccl.cu:50`
  - `/workspace/sglang-main/python/sglang/srt/distributed/parallel_state.py:663`
  - `/workspace/sglang-main/python/sglang/srt/distributed/device_communicators/npu_communicator.py:27`
  - `/workspace/sglang-main/python/sglang/srt/layers/linear.py:1546`

### projection/cache-boundary fusion

- Status: reference path exists, but lower priority for next target
- Evidence: source-derived plus owner timing
- Finding: Mini already has fused q/kv norm+RoPE+BF16 cache store and projection BF16 caches. SGLang/vLLM have DSv4 fused norm/rope/FP8 store and fused pack/store code. The current owner timing does not make this a stronger next move than FP8 KV/cache capacity work.
- Sources:
  - `python/minisgl/kernel/triton/deepseek_v4.py:3557`
  - `python/minisgl/kernel/triton/deepseek_v4.py:3644`
  - `python/minisgl/kernel/triton/deepseek_v4.py:4583`
  - `python/minisgl/kernel/triton/deepseek_v4.py:5197`
  - `/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh:42`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:382`

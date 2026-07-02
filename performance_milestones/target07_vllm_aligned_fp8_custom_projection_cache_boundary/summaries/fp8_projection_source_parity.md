# FP8 Projection Source Parity

This is a source-boundary comparison against `/workspace/vllm-dsv4-docker`.
It is not a vLLM runtime bucket profile.

## Readout

The vLLM DeepSeek V4 path supports the hypothesis that projection-adjacent
work is low-precision/custom-op shaped, but it does not expose a drop-in SM80
projection kernel that mini can safely wire into the promoted path.  On SM80,
several vLLM FP8 mechanisms either route through generic quantized linear
dispatch, Marlin-style weight-only handling, or explicit reference/dequant
fallbacks.  Mini's promoted BF16 cache already removes per-decode weight
dequantization for the main projection owners.

## Parity Table

| vLLM mechanism | vLLM source | Mini analogue | Precision/cache contract | 07.72 conclusion |
| --- | --- | --- | --- | --- |
| `DeepseekV4FP8Config` / `deepseek_v4_fp8` | `vllm/model_executor/models/deepseek_v4.py:DeepseekV4FP8Config`; `vllm/model_executor/layers/quantization/fp8.py:Fp8LinearMethod` | `python/minisgl/models/deepseek_v4.py:DSV4Linear`; `python/minisgl/kernel/deepseek_v4.py:quantized_linear_ref` | vLLM registers FP8 weights/scales and dispatches through generic quantized linear kernels. Mini keeps checkpoint FP8 weights but promoted decode uses cached BF16 dequantized weights for selected owners. | Source parity exists at the quantized-linear boundary, but focused mini direct-FP8 kernels are slower than promoted cached BF16 on A100. |
| Fused WQA/WKV projection | `DeepseekV4Attention.fused_wqa_wkv` as `MergedColumnParallelLinear(..., disable_tp=True)`; called from `DeepseekV4MLAModules.forward` | Mini `DSV4Attention.forward` q-proj with `MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT` and cached fused BF16 weight | vLLM has one quantized merged linear. Mini stores separate `wq_a` and `wkv`, then builds a fused BF16 cache for the promoted path. | Maps cleanly and was the required first representative, but direct fused FP8-weight Triton is much slower in microbench. |
| `wq_b`, `wo_b`, shared expert linears | vLLM `ColumnParallelLinear` / `RowParallelLinear` quantized dispatch | Mini `attn.q_wqb`, `attn.wo_b`, `shared_experts.gate_up_proj`, `shared_experts.down_proj` cached BF16 weight routes | vLLM uses quantized linears under `deepseek_v4_fp8`. Mini promoted path quantizes/fake-quantizes activation then uses BF16 cached weights and torch/CUTLASS/cuBLASLt GEMMs. | Candidate direct FP8-weight kernel is graph-safe but far slower than cached BF16 for every measured owner. |
| `fused_inv_rope_fp8_quant` | `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` | Mini inverse-RoPE/output projection boundary before `attn.wo_a` | vLLM non-reference path emits FP8 output plus scales for `deepseek_v4_fp8_einsum`; SM80 reference path emits FP32 then casts to FP8 in PyTorch. | Source maps to mini `wo_a`, but owner is small and the mini direct grouped FP8 kernel is slower than cached BF16 BMM. |
| `deepseek_v4_fp8_einsum` | `vllm/model_executor/layers/deepseek_v4_attention.py:deepseek_v4_fp8_einsum` | `python/minisgl/kernel/deepseek_v4.py:wo_a_grouped_projection_fallback`; `python/minisgl/kernel/triton/deepseek_v4.py:wo_a_grouped_projection_fp8` | vLLM has an FP8 einsum custom op with an SM80/ROCm dequant-and-einsum fallback. Mini has a direct FP8 grouped Triton probe and promoted cached BF16 grouped BMM. | Measured mini FP8 grouped path loses by about `4.47x` at M=4, so no runtime route is justified. |
| `fused_indexer_q_rope_quant` | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`; called by `DeepseekV4Indexer.forward` | Mini `DSV4Indexer.prepare_fp8_query`, `indexer_q_rope_fp8_fallback`, and promoted FP8 indexer cache | vLLM SM80 hybrid produces FP32 Q/weights then casts Q to FP8. Mini already has prior FP8 indexer/cache pieces in the victory bundle. | Remaining indexer projection surface is below standalone gate. It can stay context, not a new runtime implementation in 07.72. |
| Full `fp8_ds_mla` KV-cache | vLLM attention/cache dtype path and FlashMLA sparse backend | Mini DSV4 attention backend and KV/indexer caches | Full packed FP8 MLA cache contract. | Explicitly out of scope. The measured projection candidate does not require it, and no full KV-cache E2E work was started. |
| HC/router precision | vLLM MHC/router sources | Mini HC/router promoted route | vLLM HC `post/comb` is FP32-like; mini promoted HC carrier is BF16. Router logits remain FP32-like. | Out of scope by target policy. No HC/router precision change was made. |

## Stop-Relevant Source Finding

The only source-clean mini-local runtime candidate available without full
`fp8_ds_mla`, HC/router precision changes, or broad compile work is direct
FP8-weight projection against the existing checkpoint FP8 weights/scales.
That candidate was measured by
`scripts/focused_fp8_projection_cache_microbench.py` and failed the focused
runtime implementation gate.

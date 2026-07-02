# vLLM Projection Source Parity

This is a source-boundary comparison against `/workspace/vllm-dsv4-docker`,
not a vLLM runtime bucket profile.

| Mini owner/backend | Mini promoted contract | Closest vLLM boundary | Same dtype contract? | Adapt decision |
| --- | --- | --- | --- | --- |
| HC pre linear | `DeepseekV4DecoderLayer._hc_pre` calls `linear_bf16_fp32_fallback`; profile shows FP32 SGEMM plus splitK/reduce. | `vllm/model_executor/layers/mhc.py:mhc_pre` SM80 reference does BF16 residual to FP32 matmul plus fused Triton pre. | Partly. The matmul is FP32-like, but vLLM returns FP32 `post/comb` while mini promoted path keeps BF16 `post/comb`; 07.68 kept the vLLM-style HC cleanup opt-in only. | Do not reopen HC as this target. It is a sub-owner of projection/GEMM but below the single-owner gate. |
| MoE router / route projection | Mini caches gate weight as FP32 and uses `F.linear(hidden.float(), weight.float())` inside `moe_gate_fallback`. | `vllm/model_executor/models/deepseek_v4.py:DeepseekV4MoE` uses `GateLinear(..., out_dtype=torch.float32)` and passes router ownership into `FusedMoE`. | Router-logit dtype is aligned, but vLLM has a different fused-MoE/compile ownership boundary. | Name and track; do not make it the next implementation target by itself (`0.097109s`). |
| Attention WQA/WKV/compress | Mini uses promoted fused WQA/WKV shared activation plus cached BF16 dequantized weight and `F.linear`; q_proj/compress owner group is CUTLASS BF16 plus splitK. | `DeepseekV4Attention` builds `MergedColumnParallelLinear(..., quant_config=deepseek_v4_fp8)` and the MLA wrapper calls `fused_wqa_wkv(hidden_states)`. | No for the fast vLLM path. vLLM keeps the quantized FP8 linear contract, while mini promoted path materializes cached BF16 weights for exactness. | Best BF16-cluster representative for next exact backend target; precision-porting it would be a separate policy target. |
| Attention `q_wqb`, `wo_a`, `wo_b` | Mini promoted path uses cached BF16 `F.linear` for `q_wqb` and `wo_b`; `wo_a` uses cached BF16 grouped `torch.bmm`. | vLLM lifts `wq_b` out before `deepseek_v4_attention`; `wo_a` has an SM80 BF16 BMM reference path and a non-reference FP8 inverse-RoPE/einsum path; `wo_b` is `RowParallelLinear` under the quantized layer stack. | `wo_a` SM80 reference is close to mini's BF16 BMM cache. vLLM fast path is not the same contract because it uses FP8 quant/einsum before `wo_b`. | Include in BF16 small-GEMM cluster target; do not claim vLLM FP8-einsum is drop-in. |
| Indexer `wq_b` and weights/compressor | Mini uses cached BF16 `indexer.wq_b` plus FP8 indexer cache/select opt-in already in the victory bundle. | vLLM `DeepseekV4Indexer` uses quantized `ReplicatedLinear`, `fused_indexer_q_rope_quant`, and FP8/FP4 indexer cache layout. | No. vLLM uses a packed low-precision indexer/cache boundary, not mini's exact cached BF16 `wq_b` contract. | Keep as a named contributor inside the BF16 cluster; broader indexer precision/cache work is a different target. |
| Shared experts | Mini 07.66 promoted cached BF16 gate/up/down projection weights, leaving compute as BF16 GEMM/CUTLASS. | vLLM passes `DeepseekV2MLP` shared experts into `FusedMoE`; its quant config is `deepseek_v4_fp8`, with MoE routed layers overridden to MXFP4/WNA-style methods on the fast path. | No for the fast path. Ownership and precision differ. | Include only as a cluster contributor. Do not continue shared-expert staging; 07.66 already removed the direct-copy owner. |
| `lm_head` | Mini uses FP32 `F.linear(x.float(), weight.float())` plus TP gather; profile projection/GEMM owner is only `0.026769s`. | vLLM uses `ParallelLMHead` through the logits processor. | Similar high-level output head boundary, but not a selected bottleneck. | Track only. |

vLLM also uses source boundaries that are outside an exact projection/GEMM
drop-in: `DeepseekV4FP8Config` advertises `deepseek_v4_fp8`; attention requires
FP8 KV/cache format and may canonicalize to `fp8_ds_mla`; the wrapper uses
`torch.ops.vllm.deepseek_v4_attention`, `fused_inv_rope_fp8_quant`, and
`torch.ops.vllm.deepseek_v4_fp8_einsum`; the model is under
`@support_torch_compile`.  These explain why source parity is useful as a
boundary map but not sufficient to promote a vLLM-style low-precision backend
inside the current exact BF16 mini route.

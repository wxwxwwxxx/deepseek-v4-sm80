# FP8 Projection Surface Mapping

Baseline interpretation:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- page size: `256`
- TP8, 8x A100
- inactive opt-ins not used as baseline:
  `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`,
  `MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1`,
  `MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1`

## Mapping Gate

TARGET 07.69 measured the promoted projection/GEMM bucket at `0.778887s`,
with a BF16 small-GEMM plus splitK/reduce cluster of `0.521619s`.  The named
owners below sum to `0.466436s`; the remaining cluster time is parent
splitK/reduce and backend overhead attributed in the 07.69 classifier.

Mapping gate result: pass.  The vLLM-aligned FP8/custom projection-cache
candidate maps to at least `0.35s` of coherent current mini surface.

## Surface Table

| Mini owner | Current time | Current mini contract | vLLM analogue | FP8/custom candidate tested | Include? |
| --- | ---: | --- | --- | --- | --- |
| attention WQA/WKV/compress | `0.119458s` | Separate checkpoint FP8 `wq_a`/`wkv`, promoted fused BF16 dequantized cache plus FP8-style activation rounding | `DeepseekV4Attention.fused_wqa_wkv` via quantized `MergedColumnParallelLinear` | Fused original FP8 weight/scale cache plus mini Triton direct FP8-weight GEMM | Yes, required representative |
| shared experts cached BF16 | `0.085848s` | Promoted cached BF16 gate/up and down projection weights, activation rounding before GEMM | vLLM shared expert linears under `deepseek_v4_fp8` quantized dispatch | Original FP8 weights/scales with mini direct FP8-weight GEMM | Yes, required representative |
| attention `wo_a` | `0.063857s` | Promoted cached BF16 grouped BMM | `fused_inv_rope_fp8_quant` plus `deepseek_v4_fp8_einsum` | Mini direct grouped FP8-weight Triton kernel | Secondary representative |
| attention `q_wqb` | `0.056392s` | Promoted cached BF16 dequantized weight | vLLM quantized `ColumnParallelLinear` | Original FP8 weight/scale plus mini Triton direct FP8-weight GEMM | Yes, required one-of representative |
| attention `wo_b` local | `0.054507s` | Promoted cached BF16 dequantized local row-parallel projection, all-reduce separate | vLLM quantized `RowParallelLinear` | Original FP8 weight/scale plus mini Triton direct FP8-weight GEMM | Yes, required one-of representative |
| indexer weight/compressor projection | `0.043647s` | Mixed BF16 projection/compressor work around indexer path | vLLM indexer projection before fused q/rope/quant | Context only in this target | No runtime test |
| indexer `wq_b` | `0.042727s` | Promoted cached BF16 dequantized weight where layer has indexer | vLLM `DeepseekV4Indexer` projection plus `fused_indexer_q_rope_quant` | Direct FP8-weight GEMM when indexer weight exists | Optional/context |

## Coherent Surface Total

| Surface | Time |
| --- | ---: |
| Named mapped owners | `0.466436s` |
| 07.69 BF16 small-GEMM plus splitK/reduce cluster | `0.521619s` |
| Required mapping floor | `0.350000s` |
| Gate result | pass |

## Focused Gate Readout

The mapping was large enough to justify a focused real-shape microbench, but
the runtime implementation gate failed.  At M=4, no measured owner improved
by `>=15%`; all direct FP8/custom candidates regressed versus promoted cached
BF16.

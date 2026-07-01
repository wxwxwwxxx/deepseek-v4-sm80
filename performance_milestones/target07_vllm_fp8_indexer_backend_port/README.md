# TARGET 07.52: vLLM-Aligned FP8 Indexer Backend Port

Date: 2026-07-02

## Backend Contract

| Surface | mini exact default | vLLM FP8 indexer contract | Ported opt-in mini contract |
| --- | --- | --- | --- |
| Guard | BF16 path, no FP8 side cache | Model path owns FP8 indexer tensors | `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1`; exact default unchanged |
| Q | BF16 `[rows, heads, dim]` | E4M3 bytes; per-row/head Q scale folded into weights | `indexer_fp8_quantize_fold`: E4M3 bytes plus folded FP32 weights |
| K cache | BF16 C4 indexer flat cache | Per block flat `[block_size * dim bytes][block_size * 4 scale bytes]` | Packed paged uint8 cache, indexer `page_size = model_page_size / 4 = 64` |
| K scale | none | one float32 scale per token row, stored as 4 bytes | same row scale layout in packed page tail |
| Decode logits | BF16 paged MQA/indexer logits | `fp8_paged_mqa_logits_triton`, uint8 load plus BF16 LUT decode on SM80 | Triton paged logits kernel with BF16 LUT decode, ReLU, folded weights |
| Page table | mini full-token physical loc table | paged block table | mini full page ids reused; C4 page size is 64 for model page size 256 |
| Top-k | mini global top-k/lens transform | vLLM top-k downstream | unchanged mini top-k/lens after FP8 logits |

## Source Mapping

| vLLM source | mini target |
| --- | --- |
| `vllm/model_executor/layers/sparse_attn_indexer.py` Q/indexer semantics | `python/minisgl/models/deepseek_v4.py`, `python/minisgl/kernel/deepseek_v4.py` |
| `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` Q scale folding | `indexer_q_rope_fp8_fallback` -> `triton.indexer_fp8_quantize_fold` |
| `vllm/_custom_ops.py` / `csrc/cache_kernels.cu` indexer K quant/cache | `store_indexer_fp8_cache_fallback` -> `triton.indexer_fp8_paged_quant_store` |
| `vllm/v1/attention/ops/mqa_logits_triton.py` `fp8_paged_mqa_logits_triton` | `triton.indexer_fp8_paged_logits` and `indexer_fp8_paged_logits_fallback` |

Out of scope stayed out of scope: no full `fp8_ds_mla` KV cache E2E and no standalone `quantize_and_insert_k_cache` port.

## Implementation Summary

- Added packed paged FP8 indexer cache allocation in `DeepSeekV4KVCache` under `MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1`.
- Added SM80 software E4M3 encode/decode path, BF16 LUT decode, Q quant plus scale folding, paged K quant/store, and paged FP8 logits.
- Made graph capture safe by prewarming the BF16 LUT before `GraphRunner` capture and by removing Triton autotune from the paged logits capture path.
- Kept exact BF16 default allocation and selection path unchanged.
- The existing perf matrix captures CUDA graphs once at engine init, so exact and FP8 graph variants must be run as separate single-variant processes for fair graph semantics.

## Microbench Gate

Source:

- Raw: `raw/mini_vllm_fp8_indexer_backend_microbench.json`
- Summary: `summaries/mini_vllm_fp8_indexer_backend_microbench_summary.json`

| Shape | Q fold ms | K store ms | BF16 logits ms | FP8 paged logits ms | BF16 select ms | FP8 select ms | Logits speedup | Select speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1/h1024 | 0.0890 | 0.0681 | 0.1609 | 0.1794 | 0.2154 | 0.2478 | 0.90x | 0.87x |
| b4/h2048 | 0.0879 | 0.0668 | 0.1655 | 0.1791 | 0.2139 | 0.2449 | 0.92x | 0.87x |
| b16/h4096 | 0.0870 | 0.1215 | 0.3516 | 0.1845 | 0.3709 | 0.2472 | 1.91x | 1.50x |

Large-shape comparison to TARGET 07.51 vLLM:

- Q fold: `0.0870 ms`, `1.04x` vLLM `0.0839 ms`.
- K store: `0.1215 ms`, `1.26x` vLLM `0.0964 ms`.
- Paged logits: `0.1845 ms`, `1.21x` vLLM `0.1529 ms`, within the `1.25x` gate.
- Logits plus select beats mini BF16 select by `1.50x`, above the required 15% win.

## Quality Gate

| Shape | Logits mean abs | Logits max abs | Top-k overlap mean/min | K dequant mean/max abs |
| --- | ---: | ---: | ---: | ---: |
| b1/h1024 | 0.02263 | 0.10611 | 0.9902 / 0.9902 | 0.01796 / 0.1875 |
| b4/h2048 | 0.02101 | 0.11128 | 0.9785 / 0.9766 | 0.01796 / 0.25 |
| b16/h4096 | 0.02147 | 0.14102 | 0.9744 / 0.9648 | 0.01796 / 0.25 |

Text smoke:

- `raw/text_smoke_idxfp8cache_fixedlaunch.json`
- Status: pass.
- Graph: captured `[4, 2, 1]`, replay count `9`, eager decode count `0`.

## Macro Gate

Discarded runs:

- `raw/macro_4096x128_bs4`: default page allocation OOM, not a gate artifact.
- Two-variant FP8-second runs: invalid for graph semantics because the runner captures one graph at engine init.

Gate artifacts:

| Run | Variant | Output tok/s | Decode tok/s | Graph replay | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `raw/macro_4096x128_bs4_np128` | exact control | 31.93 | 79.68 | 127 | pass |
| `raw/macro_4096x128_bs4_np128_fp8only_fixedlaunch` | FP8 indexer | 41.63 | 85.76 | 127 | pass |
| `raw/macro_4096x1024_bs4_np128_fp8only_fixedlaunch` | FP8 indexer | 73.67 | 86.21 | 1023 | pass |

4096/128 gate: FP8 improves output throughput by `30.4%` over this graph-correct exact control, and decode throughput by `7.6%`.

4096/1024: FP8 reaches `73.67 output tok/s`, above the target file's exact reference `68.81` by `7.1%`, but still below the old serving victory line `114.07` and far below fresh vLLM `~202`.

## Final Decision

Decision: promote opt-in vLLM-aligned FP8 indexer and reprofile macro

Do-not-continue condition: do not resume local polishing of the old mini-owned FP8 indexer slice, and do not port standalone `quantize_and_insert_k_cache`. The remaining gap is no longer the isolated indexer logits backend; next work should reprofile full macro bottlenecks with this opt-in path enabled and then choose between graph/layout integration work and the real vLLM fused compressor/insert SM80 software-FP8 KV-store path.

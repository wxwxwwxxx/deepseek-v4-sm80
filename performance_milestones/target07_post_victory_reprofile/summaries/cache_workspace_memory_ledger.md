# Post-Victory Cache/Workspace Memory Ledger

- report: `/tmp/dsv4_target0763_4096x128_bs4_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory.json`
- baseline report: `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/macro_qwqb_wob_idxwqb_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`

| Owner | Enabled | Layers | Shape/rank | Dtype | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank | Lifecycle | Ownership |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `attn.q_wqb` | `True` | 43 | `[4096, 1024]` | `torch.bfloat16` | 360,710,144 | `0.3359` | `4744.04` | `18.53` | prebuilt before graph capture | owned by model module cache attribute |
| `attn.wo_b` | `True` | 43 | `[4096, 1024]` | `torch.bfloat16` | 360,710,144 | `0.3359` | `4744.04` | `18.53` | prebuilt before graph capture | owned by model module cache attribute |
| `indexer.wq_b` | `True` | 21 | `[8192, 1024]` | `torch.bfloat16` | 352,321,536 | `0.3281` | `4633.71` | `18.10` | prebuilt before graph capture | owned by model module cache attribute |
| `attn.wo_a` | `True` | 43 | `[1, 4096, 1024]` | `torch.bfloat16` | 360,710,144 | `0.3359` | `4744.04` | `18.53` | prebuilt before graph capture | owned by model module cache attribute |
| `indexer.fp8_paged_cache` | `True` | 21 | `[21, 128, 8448]` | `torch.uint8` | 22,708,224 | `0.0211` | `298.66` | `1.17` | allocated with KV cache pool before decode; populated during prefill/decode store | currently owned by DeepSeekV4KVCache; included in kv_cache_memory_bytes_per_rank_max |
| `moe_v2_workspace` | `False` | 0 | `lazy reusable buffers` | `mixed` | 0 | `0.0000` | `0.00` | `0.00` | not materialized in current Marlin WNA16 MoE backend | ad hoc DSV4MoEWorkspace exists for grouped backend; inactive here |

| Total | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank |
| --- | ---: | ---: | ---: | ---: |
| `cached_bf16_projection` | 1,434,451,968 | `1.3359` | `18865.83` | `73.69` |
| `listed_extra_cache_and_workspace` | 1,457,160,192 | `1.3571` | `19164.48` | `74.86` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs baseline | `360,710,144` |
| peak reserved delta vs baseline | `381,681,664` |
| graph replay count | `508` |
| eager decode count | `0` |

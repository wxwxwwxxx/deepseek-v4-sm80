# wo_a BF16 BMM Cache Memory Ledger

- wo_a report: `performance_milestones/target07_wo_a_attention_boundary_parity/raw/macro_wo_a_bf16_bmm_cache_4096x128_bs4_np128/reports/000_decode_throughput_bs8__target0762_woabf16bmmcache.json`
- baseline report: `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/macro_qwqb_wob_idxwqb_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`

| Cached owner | Enabled | Layers | Cache shape/rank | Source shape/rank | Extra bytes/rank | Extra GiB/rank | KV tokens/rank | KV pages/rank |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `indexer.wq_b` | `True` | 21 | `[8192, 1024]` | `mixed` | 352,321,536 | `0.3281` | `4633.71` | `18.10` |
| `attn.wo_a` | `True` | 43 | `[1, 4096, 1024]` | `[1024, 4096]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `total_cached_bf16_projection` | `n/a` | 0 | `mixed` | `mixed` | 1,434,451,968 | `1.3359` | `18865.83` | `73.69` |
| `attn.wo_a` | `n/a` | 0 | `mixed` | `mixed` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs baseline | `360,710,144` |
| peak reserved delta vs baseline | `381,681,664` |
| graph replay count | `127` |
| eager decode count | `0` |

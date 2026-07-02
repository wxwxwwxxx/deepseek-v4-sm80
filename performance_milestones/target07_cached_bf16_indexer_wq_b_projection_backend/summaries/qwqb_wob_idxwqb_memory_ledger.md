# q_wqb + wo_b + indexer.wq_b Cached BF16 Memory Ledger

- q_wqb + wo_b + indexer.wq_b report: `performance_milestones/target07_cached_bf16_indexer_wq_b_projection_backend/raw/macro_qwqb_wob_idxwqb_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`
- q_wqb + wo_b baseline report: `performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/macro_qwqb_wob_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`
- exact baseline report: `performance_milestones/target07_graph_layout_replay_deforestation/raw/macro_4096x128_bs4_np128_actqtriton/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `attn.wo_b` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |
| `indexer.wq_b` | 21 | `[8192, 1024]` | 352,321,536 | `0.3281` | `4633.71` | `18.10` |
| `total` | 107 | `mixed` | 1,073,741,824 | `1.0000` | `14121.79` | `55.16` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs q_wqb+wo_b baseline | `351,272,960` |
| peak reserved delta vs q_wqb+wo_b baseline | `352,321,536` |
| peak allocated delta vs exact baseline | `1,077,149,696` |
| peak reserved delta vs exact baseline | `1,214,251,008` |
| graph replay count | `127` |
| eager decode count | `0` |

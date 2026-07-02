# q_wqb Cached BF16 Memory Ledger

- q_wqb report: `performance_milestones/target07_cached_bf16_projection_backend/raw/macro_qwqb_4096x128_bs4_np128/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`
- baseline report: `performance_milestones/target07_graph_layout_replay_deforestation/raw/macro_4096x128_bs4_np128_actqtriton/reports/000_decode_throughput_bs8__v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache.json`

| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | 43 | `[4096, 1024]` | 360,710,144 | `0.3359` | `4744.04` | `18.53` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `76034.41` |
| page size | `256` |
| num pages | `128` |
| KV cache bytes/rank max | `2,491,495,680` |
| peak allocated delta vs baseline | `363,565,056` |
| peak reserved delta vs baseline | `417,333,248` |
| graph replay count | `127` |
| eager decode count | `0` |

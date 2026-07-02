# wo_a BF16 BMM Cache Memory Ledger

- wo_a report: `performance_milestones/target07_wo_a_attention_boundary_parity/raw/text_smoke_wo_a_bf16_bmm_cache.target0762_woabf16bmmcache.json`

| Cached owner | Enabled | Layers | Cache shape/rank | Source shape/rank | Extra bytes/rank | Extra GiB/rank | KV tokens/rank | KV pages/rank |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `n/a` | `n/a` |
| `attn.wo_b` | `True` | 43 | `[4096, 1024]` | `mixed` | 360,710,144 | `0.3359` | `n/a` | `n/a` |
| `indexer.wq_b` | `True` | 21 | `[8192, 1024]` | `mixed` | 352,321,536 | `0.3281` | `n/a` | `n/a` |
| `attn.wo_a` | `True` | 43 | `[1, 4096, 1024]` | `[1024, 4096]` | 360,710,144 | `0.3359` | `n/a` | `n/a` |
| `total_cached_bf16_projection` | `n/a` | 0 | `mixed` | `mixed` | 1,434,451,968 | `1.3359` | `n/a` | `n/a` |
| `attn.wo_a` | `n/a` | 0 | `mixed` | `mixed` | 360,710,144 | `0.3359` | `n/a` | `n/a` |

| Metric | Value |
| --- | ---: |
| bytes/token/rank | `n/a` |
| page size | `256` |
| num pages | `64` |
| KV cache bytes/rank max | `0` |
| peak allocated delta vs baseline | `n/a` |
| peak reserved delta vs baseline | `n/a` |
| graph replay count | `9` |
| eager decode count | `0` |

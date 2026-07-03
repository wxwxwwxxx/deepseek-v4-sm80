# Marlin Projection Memory Ledger

- created_at: `2026-07-03 00:11:09`
- num_layers: `43`
- page_size: `256`
- KV-cache bytes/page/rank: `19313920`

## Per Owner

| Owner | BF16 cache/layer | Original FP8/layer | Marlin weight+scale/layer | Workspace/layer | Marlin total/layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention q_wqb | `8388608` | `4194560` | `4259840` | `432` | `4260272` |
| attention wo_b local | `8388608` | `4194560` | `4259840` | `432` | `4260272` |
| shared experts down | `2097152` | `1048640` | `1064960` | `432` | `1065392` |

## Totals Per Rank

| Metric | Bytes | GiB | KV pages @256 | KV tokens @256 |
| --- | ---: | ---: | ---: | ---: |
| Promoted BF16 cache incremental | `811597824` | `0.7559` | `42.02` | `10757` |
| Promoted original FP8 weight+scale | `405823680` | `0.3780` | `21.01` | `5379` |
| Promoted owner total if original retained | `1217421504` | `1.1338` | `63.03` | `16137` |
| Marlin weight+scale | `412139520` | `0.3838` | `21.34` | `5463` |
| Marlin workspace | `55728` | `0.0001` | `0.00` | `1` |
| Marlin total after release | `412195248` | `0.3839` | `21.34` | `5464` |
| Released original FP8 weight+scale | `405823680` | `0.3780` | `21.01` | `5379` |
| Delta vs BF16 cache incremental | `-399402576` | `-0.3720` | `-20.68` | `-5294` |
| Delta vs promoted owner total | `-805226256` | `-0.7499` | `-41.69` | `-10673` |

Interpretation: Marlin total includes packed weight, expanded/permuted scale, and vLLM workspace. The runtime opt-in releases original FP8 weight/scale for switched owners after successful pack; no BF16 cache is built for q_wqb, wo_b, or shared-down under this toggle.

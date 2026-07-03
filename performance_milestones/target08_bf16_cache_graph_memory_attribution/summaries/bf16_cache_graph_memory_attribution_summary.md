# TARGET 08.07 BF16 Cache Graph Memory Attribution Summary

All GiB values use bytes / 2^30.

## Single-Bucket Attribution

| run | buckets | denylist | enabled q/woB/woA/idx/shared | persistent GiB | free before/after/delta GiB | alloc delta GiB | reserved delta GiB | vs baseline GiB | replay/eager |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `single_full_victory` | `[16]` | `` | `Y/Y/Y/Y/Y` | 1.588 | 55.485/36.657/18.828 | 17.820 | 18.191 | 0.000 | 63/0 |
| `single_no_projection_bf16_caches` | `[16]` | `projection_bf16_caches` | `N/N/N/N/Y` | 0.252 | 56.821/37.942/18.879 | 17.821 | 18.234 | 0.051 | 63/0 |
| `single_no_q_wqb_bf16_cache` | `[16]` | `q_wqb` | `N/Y/Y/Y/Y` | 1.252 | 55.833/37.013/18.820 | 17.820 | 18.180 | -0.008 | 63/0 |
| `single_no_wo_b_bf16_cache` | `[16]` | `wo_b` | `Y/N/Y/Y/Y` | 1.252 | 55.833/37.009/18.824 | 17.820 | 18.184 | -0.004 | 63/0 |
| `single_no_wo_a_bf16_bmm_cache` | `[16]` | `wo_a` | `Y/Y/N/Y/Y` | 1.252 | 55.829/37.005/18.824 | 17.820 | 18.184 | -0.004 | 63/0 |
| `single_no_indexer_wq_b_bf16_cache` | `[16]` | `indexer_wq_b` | `Y/Y/Y/N/Y` | 1.260 | 55.817/36.989/18.828 | 17.821 | 18.188 | 0.000 | 63/0 |
| `single_no_shared_expert_bf16_cache` | `[16]` | `shared_expert` | `Y/Y/Y/Y/N` | 1.336 | 55.599/36.749/18.850 | 17.820 | 18.207 | 0.021 | 63/0 |
| `single_no_all_tested_bf16_caches` | `[16]` | `all_tested_bf16_caches` | `N/N/N/N/N` | 0.000 | 57.077/38.192/18.885 | 17.820 | 18.234 | 0.057 | 63/0 |

## Full-Bucket Confirmation

| run | buckets | denylist | enabled q/woB/woA/idx/shared | persistent GiB | free before/after/delta GiB | alloc delta GiB | reserved delta GiB | vs baseline GiB | replay/eager |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `full_full_victory` | `[1, 2, 4, 8, 16]` | `` | `Y/Y/Y/Y/Y` | 1.588 | 55.485/36.448/19.037 | 17.820 | 18.191 | 0.000 | 63/0 |

## Cache Owner Matrix

| run | disabled toggles | owner bytes GiB | enabled owners |
| --- | --- | --- | --- |
| `single_full_victory` | `[]` | q_wqb:0.336<br>wo_b:0.336<br>wo_a:0.336<br>indexer_wq_b:0.328<br>shared_expert:0.252 | `['attn.q_wqb', 'attn.wo_b', 'indexer.wq_b', 'attn.wo_a', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_projection_bf16_caches` | `['MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE', 'MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE', 'MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE', 'MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE', 'MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE']` | q_wqb:0.000<br>wo_b:0.000<br>wo_a:0.000<br>indexer_wq_b:0.000<br>shared_expert:0.252 | `['shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_q_wqb_bf16_cache` | `['MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE']` | q_wqb:0.000<br>wo_b:0.336<br>wo_a:0.336<br>indexer_wq_b:0.328<br>shared_expert:0.252 | `['attn.wo_b', 'indexer.wq_b', 'attn.wo_a', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_wo_b_bf16_cache` | `['MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE']` | q_wqb:0.336<br>wo_b:0.000<br>wo_a:0.336<br>indexer_wq_b:0.328<br>shared_expert:0.252 | `['attn.q_wqb', 'indexer.wq_b', 'attn.wo_a', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_wo_a_bf16_bmm_cache` | `['MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE']` | q_wqb:0.336<br>wo_b:0.336<br>wo_a:0.000<br>indexer_wq_b:0.328<br>shared_expert:0.252 | `['attn.q_wqb', 'attn.wo_b', 'indexer.wq_b', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_indexer_wq_b_bf16_cache` | `['MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE']` | q_wqb:0.336<br>wo_b:0.336<br>wo_a:0.336<br>indexer_wq_b:0.000<br>shared_expert:0.252 | `['attn.q_wqb', 'attn.wo_b', 'attn.wo_a', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |
| `single_no_shared_expert_bf16_cache` | `['MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE']` | q_wqb:0.336<br>wo_b:0.336<br>wo_a:0.336<br>indexer_wq_b:0.328<br>shared_expert:0.000 | `['attn.q_wqb', 'attn.wo_b', 'indexer.wq_b', 'attn.wo_a']` |
| `single_no_all_tested_bf16_caches` | `['MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE', 'MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE', 'MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE', 'MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE', 'MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE', 'MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE']` | q_wqb:0.000<br>wo_b:0.000<br>wo_a:0.000<br>indexer_wq_b:0.000<br>shared_expert:0.000 | `[]` |
| `full_full_victory` | `[]` | q_wqb:0.336<br>wo_b:0.336<br>wo_a:0.336<br>indexer_wq_b:0.328<br>shared_expert:0.252 | `['attn.q_wqb', 'attn.wo_b', 'indexer.wq_b', 'attn.wo_a', 'shared_experts.gate_up_proj', 'shared_experts.down_proj']` |

## Materiality

- Phase-2 threshold: > 1.000 GiB/rank.
- Small-fix threshold: graph-delta reduction > 2.000 GiB/rank.
- Phase-2 candidates from single-bucket data: `[]`.
- Small-fix candidates from single-bucket data: `[]`.

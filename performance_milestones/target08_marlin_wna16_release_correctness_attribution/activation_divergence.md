# Activation Divergence (Eager max_tokens=4)

Same-sign nonfinite values are treated as equal; `inf` max diff means finite/nonfinite pattern mismatch.
First divergence: `layer2.indexer_select.logits` activation_index=3486 max_abs_diff=inf mean_abs_diff=0.0

| Index | Name | Phase | Req lens | Max abs diff | Mean abs diff | Nonfinite mismatch |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 3486 | `layer2.indexer_select.logits` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | inf | 0.0 | 2 |
| 3491 | `layer2.indexer_select.topk_scores` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | inf | 0.0 | 2 |
| 3497 | `layer2.attention_backend.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 9.437518770072903e+37 | inf | 0 |
| 3498 | `layer2.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 9.437518770072903e+37 | inf | 0 |
| 3499 | `layer2.merged_attention_output_after_inverse_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 9.437518770072903e+37 | inf | 0 |
| 3500 | `layer2.final_attention_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.9046072376181904e+37 | inf | 0 |
| 3501 | `layer2.attention_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.9046072376181904e+37 | inf | 0 |
| 3502 | `layer2.moe_input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.2265625 | 0.16906189918518066 | 0 |
| 3503 | `layer2.moe_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.91796875 | 0.09434259682893753 | 0 |
| 3504 | `layer3.input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.738453738145076e+36 | inf | 0 |
| 3505 | `layer3.attention_input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.21875 | 0.024034710600972176 | 0 |
| 3506 | `layer3.wqa_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.435546875 | 0.06781626492738724 | 0 |
| 3507 | `layer3.wkv_shared_activation_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.88671875 | 0.06730352342128754 | 0 |
| 3508 | `layer3.q_lora_after_norm` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.1083984375 | 0.01554225292056799 | 0 |
| 3509 | `layer3.q_wqb_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.375 | 0.011978454887866974 | 0 |
| 3510 | `layer3.wkv_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.88671875 | 0.06730352342128754 | 0 |
| 3511 | `layer3.q_after_q_norm_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 11.375 | 0.45273593068122864 | 0 |
| 3512 | `layer3.kv_after_kv_norm_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 6.65625 | 0.35416126251220703 | 0 |
| 3518 | `layer3.attention_backend.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 5.626953125 | 0.12371978908777237 | 0 |
| 3519 | `layer3.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 5.626953125 | 0.12371978908777237 | 0 |
| 3520 | `layer3.merged_attention_output_after_inverse_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 5.625 | 0.12399452179670334 | 0 |
| 3521 | `layer3.final_attention_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 7.0625 | 0.716289222240448 | 0 |
| 3522 | `layer3.attention_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 7.0625 | 0.716289222240448 | 0 |
| 3523 | `layer3.moe_input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.0859375 | 0.14133402705192566 | 0 |
| 3524 | `layer3.moe_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.54296875 | 0.07329286634922028 | 0 |
| 3525 | `layer4.input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.6346078009743793e+36 | inf | 0 |
| 3526 | `layer4.attention_input` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 2.703125 | 0.027113107964396477 | 0 |
| 3527 | `layer4.wqa_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.5390625 | 0.07025016099214554 | 0 |
| 3528 | `layer4.wkv_shared_activation_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.359375 | 0.052805397659540176 | 0 |
| 3529 | `layer4.q_lora_after_norm` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.125 | 0.01579318940639496 | 0 |
| 3530 | `layer4.q_wqb_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.515625 | 0.01364626083523035 | 0 |
| 3531 | `layer4.wkv_output` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.359375 | 0.052805397659540176 | 0 |
| 3532 | `layer4.q_after_q_norm_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 10.875 | 0.46900805830955505 | 0 |
| 3533 | `layer4.kv_after_kv_norm_rope` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 4.09375 | 0.1733940988779068 | 0 |
| 3534 | `layer4.indexer_query_fp8_values` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 254.0 | 105.2476806640625 | 0 |
| 3535 | `layer4.indexer_query_fp8_weights` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 0.0001267777115572244 | 2.5151713998639025e-05 | 0 |
| 3539 | `layer4.indexer_select.logits` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.031945824623108 | 0.22149218618869781 | 0 |
| 3544 | `layer4.indexer_select.topk_scores` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 1.031945824623108 | 0.22149218618869781 | 0 |
| 3550 | `layer4.attention_backend.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.0323013653843393e+36 | 5.1705131371485326e+33 | 0 |
| 3551 | `layer4.merged_attention_output_before_wo` | decode | uid0:31/32, uid1:34/35, uid2:27/28 | 3.0323013653843393e+36 | 5.1705131371485326e+33 | 0 |

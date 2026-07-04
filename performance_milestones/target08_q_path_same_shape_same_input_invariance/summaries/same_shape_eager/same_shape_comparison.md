| Phase | Checkpoint | Shape change bs1->bs4 | Same-shape target slot | Same-shape filler content | Identical rows |
| --- | --- | --- | --- | --- | --- |
| prefill | layer0.attention_input | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.wqa_output | pass max=0.000976562 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_lora_after_norm | pass max=0.000488281 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_wqb_output | pass max=0.00195312 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.q_after_q_norm_rope | FAIL max=0.0351562 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| prefill | layer0.final_attention_output | FAIL max=0.0625 | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.attention_input | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.wqa_output | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_lora_after_norm | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_wqb_output | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.q_after_q_norm_rope | pass max=0, exact | pass max=0, exact | pass max=0, exact | pass max=0, exact |
| decode | layer0.final_attention_output | FAIL max=0.078125 | pass max=0, exact | pass max=0, exact | pass max=0, exact |

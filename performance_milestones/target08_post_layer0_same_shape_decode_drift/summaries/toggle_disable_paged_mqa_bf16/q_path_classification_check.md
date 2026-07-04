| Phase | Checkpoint | Target-slot | Filler-content | Identical-row |
| --- | --- | --- | --- | --- |
| prefill | layer0.attention_input | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.wqa_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.q_lora_after_norm | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.q_wqb_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.q_after_q_norm_rope | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| prefill | layer0.final_attention_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.attention_input | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.wqa_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.q_lora_after_norm | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.q_wqb_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.q_after_q_norm_rope | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode0 | layer0.final_attention_output | exact max=0 mean=0 | exact max=0 mean=0 | exact max=0 mean=0 |
| decode1 | layer0.attention_input | diff max=0.0908203 mean=0.017193 | exact max=0 mean=0 | diff max=0.130371 mean=0.0277975 |
| decode1 | layer0.wqa_output | diff max=0.185791 mean=0.0411628 | exact max=0 mean=0 | diff max=0.30957 mean=0.0699366 |
| decode1 | layer0.q_lora_after_norm | diff max=0.0838623 mean=0.0192671 | exact max=0 mean=0 | diff max=0.146484 mean=0.0333066 |
| decode1 | layer0.q_wqb_output | diff max=0.226562 mean=0.0161078 | exact max=0 mean=0 | diff max=0.298828 mean=0.0273285 |
| decode1 | layer0.q_after_q_norm_rope | diff max=3.70312 mean=0.439972 | exact max=0 mean=0 | diff max=8.07812 mean=0.727525 |
| decode1 | layer0.final_attention_output | diff max=3.23047 mean=0.742051 | exact max=0 mean=0 | diff max=6.65625 mean=1.22831 |

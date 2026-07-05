| bs | selected rows | BF16 combined ms | FP8 separated combined ms | delta ms | graph workspace |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 0.044858 | 0.060692 | +0.015833 | 0.12 |
| 2 | 256 | 0.042947 | 0.060492 | +0.017545 | 0.25 |
| 4 | 512 | 0.044360 | 0.061223 | +0.016863 | 0.50 |
| 8 | 1024 | 0.044981 | 0.061709 | +0.016728 | 1.00 |
| 16 | 2048 | 0.045042 | 0.061205 | +0.016163 | 2.00 |

| design | expected latency delta | evidence | risk note |
| --- | --- | --- | --- |
| BF16 baseline | 0 | runtime-proven promoted path | none |
| separated FP8 store + selected gather/dequant | +0.016 to +0.018 ms/boundary; worst 0.75 ms if paid by all layers | runtime-proven slice | too slow as production shape |
| SGLang-aligned fused store + selected-row gather/dequant | +0.006 to +0.012 ms/boundary estimate | estimated from removing store launch and keeping selected-row dequant | acceptable only as capacity opt-in until macro-proven |
| attention-integrated dequant | 0 to +0.006 ms/boundary estimate | source-derived plausible, not mini-proven | highest kernel and correctness risk |

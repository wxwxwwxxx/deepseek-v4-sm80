# wo_a BF16 BMM Cache Microbench

- Model path: `/models/DeepSeek-V4-Flash`
- Layer: `0`
- CUDA device: `NVIDIA A100-SXM4-80GB`
- TP shard: rank `0` / size `8`
- M values: `[1, 4, 8, 16]`
- Warmup/iters: `10` / `50`
- Scope: current `wo_a_grouped_projection_fallback` vs replay-time cached BF16 `torch.bmm`; cache build is reported separately.

| M | Groups | K/group | Rank | fallback total ms | cache build ms | cached BMM total ms | BMM only ms | speedup | improvement | max abs err | max rel err | ok 5e-2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 4096 | 1024 | `0.1679` | `0.1606` | `0.0285` | `0.0167` | `5.89x` | `83.02%` | `0.031250` | `2.207547` | `True` |
| 4 | 1 | 4096 | 1024 | `0.1755` | `0.1608` | `0.0292` | `0.0177` | `6.01x` | `83.36%` | `0.000000` | `0.000000` | `True` |
| 8 | 1 | 4096 | 1024 | `0.1757` | `0.1610` | `0.0287` | `0.0173` | `6.11x` | `83.64%` | `0.001953` | `0.004464` | `True` |
| 16 | 1 | 4096 | 1024 | `0.1768` | `0.1610` | `0.0287` | `0.0171` | `6.15x` | `83.75%` | `0.015625` | `0.005952` | `True` |

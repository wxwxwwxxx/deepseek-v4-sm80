# q_wqb Cached BF16 Projection Microbench

- Model path: `/models/DeepSeek-V4-Flash`
- CUDA device: `NVIDIA A100-SXM4-80GB`
- TP shard: rank `0` / size `8`
- M values: `[1, 4, 8, 16]`
- Warmup/iters: `10` / `50`

| Owner | M | K | N | current wrapper ms | current intrinsic ms | fallback dequant ms | cached BF16 F.linear ms | cached total ms | speedup vs wrapper | max abs err | max rel err | ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `attn.q_wqb.layer0` | 1 | 1024 | 4096 | `0.1300` | `0.0914` | `0.1982` | `0.0171` | `0.0776` | `1.68x` | `0.000000` | `0.000000` | `True` |
| `attn.q_wqb.layer0` | 4 | 1024 | 4096 | `0.1288` | `0.0914` | `0.1997` | `0.0176` | `0.0787` | `1.64x` | `0.000977` | `0.005780` | `True` |
| `attn.q_wqb.layer0` | 8 | 1024 | 4096 | `0.1295` | `0.0916` | `0.2031` | `0.0176` | `0.0792` | `1.63x` | `0.000008` | `0.007634` | `True` |
| `attn.q_wqb.layer0` | 16 | 1024 | 4096 | `0.1308` | `0.0917` | `0.1996` | `0.0179` | `0.0793` | `1.65x` | `0.001953` | `0.004425` | `True` |

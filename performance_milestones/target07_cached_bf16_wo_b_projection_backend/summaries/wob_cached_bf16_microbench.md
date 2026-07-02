# wo_b Cached BF16 Projection Microbench

- Model path: `/models/DeepSeek-V4-Flash`
- CUDA device: `NVIDIA A100-SXM4-80GB`
- TP shard: rank `0` / size `8`
- M values: `[1, 4, 8, 16]`
- Warmup/iters: `10` / `50`
- Scope: local row-parallel projection only; all-reduce is not included here.

| Owner | M | K | N | current FP8 wrapper ms | current intrinsic ms | fallback dequant ms | cached BF16 local F.linear ms | cached total local projection ms | speedup vs wrapper | max abs err | max rel err | ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `attn.wo_b.layer0` | 1 | 1024 | 4096 | `0.1290` | `0.0913` | `0.1920` | `0.0168` | `0.0753` | `1.71x` | `0.000000` | `0.000000` | `True` |
| `attn.wo_b.layer0` | 4 | 1024 | 4096 | `0.1272` | `0.0914` | `0.1960` | `0.0179` | `0.0781` | `1.63x` | `0.000488` | `0.007692` | `True` |
| `attn.wo_b.layer0` | 8 | 1024 | 4096 | `0.1268` | `0.0915` | `0.1951` | `0.0177` | `0.0791` | `1.60x` | `0.000000` | `0.000000` | `True` |
| `attn.wo_b.layer0` | 16 | 1024 | 4096 | `0.1238` | `0.0916` | `0.1917` | `0.0174` | `0.0768` | `1.61x` | `0.007812` | `0.007752` | `True` |

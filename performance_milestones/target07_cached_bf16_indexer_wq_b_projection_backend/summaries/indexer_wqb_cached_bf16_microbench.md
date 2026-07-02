# indexer.wq_b Cached BF16 Projection Microbench

- Model path: `/models/DeepSeek-V4-Flash`
- CUDA device: `NVIDIA A100-SXM4-80GB`
- TP shard: rank `0` / size `8`
- Checkpoint indexer layers: `[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]`
- Checkpoint indexer layer count: `21`
- M values: `[1, 4, 8, 16]`
- Warmup/iters: `10` / `50`
- Scope: replicated indexer query projection only; no all-reduce is involved.

| Owner | M | K | N | current FP8 wrapper ms | current intrinsic ms | fallback dequant ms | cached BF16 local F.linear ms | cached total local projection ms | speedup vs wrapper | max abs err | max rel err | ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `indexer.wq_b.layer2` | 1 | 1024 | 8192 | `0.1799` | `0.1683` | `0.2262` | `0.0173` | `0.0772` | `2.33x` | `0.000000` | `0.000000` | `True` |
| `indexer.wq_b.layer2` | 4 | 1024 | 8192 | `0.2570` | `0.1683` | `0.2276` | `0.0191` | `0.0776` | `3.31x` | `0.003906` | `0.003953` | `True` |
| `indexer.wq_b.layer2` | 8 | 1024 | 8192 | `0.1794` | `0.1686` | `0.2281` | `0.0175` | `0.0776` | `2.31x` | `0.031250` | `0.007692` | `True` |
| `indexer.wq_b.layer2` | 16 | 1024 | 8192 | `0.1794` | `0.1678` | `0.2291` | `0.0176` | `0.0770` | `2.33x` | `0.015625` | `0.006173` | `True` |

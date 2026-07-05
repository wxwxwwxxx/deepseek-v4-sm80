# MoE Marlin WNA16 Lifecycle Summary

## Theoretical Ledger

| Item | Bytes/rank | GiB/rank | KV pages | KV tokens |
| --- | ---: | ---: | ---: | ---: |
| `per_layer.raw_packed_w13` | 268,435,456 | 0.2500 | 5.84 | 1495 |
| `per_layer.raw_packed_w2` | 134,217,728 | 0.1250 | 2.92 | 747 |
| `per_layer.raw_w13_scale` | 16,777,216 | 0.0156 | 0.36 | 93 |
| `per_layer.raw_w2_scale` | 8,388,608 | 0.0078 | 0.18 | 47 |
| `per_layer.raw_total` | 427,819,008 | 0.3984 | 9.30 | 2382 |
| `per_layer.repacked_w13` | 268,435,456 | 0.2500 | 5.84 | 1495 |
| `per_layer.repacked_w2` | 134,217,728 | 0.1250 | 2.92 | 747 |
| `per_layer.repacked_w13_scale` | 16,777,216 | 0.0156 | 0.36 | 93 |
| `per_layer.repacked_w2_scale` | 8,388,608 | 0.0078 | 0.18 | 47 |
| `per_layer.repacked_total` | 427,819,008 | 0.3984 | 9.30 | 2382 |
| `all_layers.raw_total` | 18,396,217,344 | 17.1328 | 400.10 | 102426 |
| `all_layers.repacked_total` | 18,396,217,344 | 17.1328 | 400.10 | 102426 |
| `all_layers.raw_plus_repacked_total` | 36,792,434,688 | 34.2656 | 800.20 | 204852 |

## Parsed Raw Logs

- marlin rows: `1032`
- marlin repacked total: `441509216256` bytes
- warmup rows: `43104`
- graph stage rows: `320`

## Runs

| Run | Marlin rank0 repacked GiB | Warmup rank0 alloc delta GiB | Graph-capture rank0 alloc delta GiB |
| --- | ---: | ---: | ---: |
| `current_marlin_bs16` | 17.1328 | 17.8126 | 0.0000 |
| `forced_grouped_fp4_bs16` | - | 0.7225 | 0.0000 |
| `prebuild_marlin_bs16` | 17.1328 | 0.6798 | 0.0000 |
| `prebuild_release_marlin_bs16_decode2` | 17.1328 | 0.6798 | 0.0000 |

# Direct-Copy Owner Attribution: subboundary_summary

- total direct_copy: `0.736769s` / `191622` kernels
- named owner coverage: `0.00%`
- residual: `0.736769s` (`100.00%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs4:padded4` | `0.478802` | 119887 | `64.99%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |
| `residual coarse benchmark envelope: batch_forward_enqueue:decode:bs4:padded4` | `0.251120` | 69089 | `34.08%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |
| `residual coarse benchmark envelope: batch_prepare:decode:bs4` | `0.006847` | 2646 | `0.93%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs4:padded4` | `0.478802` | `64.99%` | narrow direct-copy NVTX around the source boundary |
| `residual coarse benchmark envelope: batch_forward_enqueue:decode:bs4:padded4` | `0.251120` | `34.08%` | narrow direct-copy NVTX around the source boundary |
| `residual coarse benchmark envelope: batch_prepare:decode:bs4` | `0.006847` | `0.93%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Pre-instrumentation control built from existing sub-boundary summary because raw sqlite is not present in this workspace.
- Coarse batch_forward owners are intentionally residual for the 07.65 owner gate.

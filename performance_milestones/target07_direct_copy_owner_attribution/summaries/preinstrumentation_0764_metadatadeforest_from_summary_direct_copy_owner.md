# Direct-Copy Owner Attribution: subboundary_summary

- total direct_copy: `0.731834s` / `189732` kernels
- named owner coverage: `0.00%`
- residual: `0.731834s` (`100.00%`)

## Direct-Copy Owner Table

| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs4:padded4` | `0.471334` | 118072 | `64.40%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |
| `residual coarse benchmark envelope: batch_forward_enqueue:decode:bs4:padded4` | `0.258662` | 70904 | `35.34%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |
| `residual coarse benchmark envelope: batch_prepare:decode:bs4` | `0.001838` | 756 | `0.25%` | `benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward` | existing sub-boundary JSON owner_breakdown; coarse benchmark NVTX; pre-instrumentation only |

## Residual Table

| Residual owner | Kernel s | Share | Needed NVTX |
| --- | ---: | ---: | --- |
| `residual coarse benchmark envelope: batch_forward:decode:bs4:padded4` | `0.471334` | `64.40%` | narrow direct-copy NVTX around the source boundary |
| `residual coarse benchmark envelope: batch_forward_enqueue:decode:bs4:padded4` | `0.258662` | `35.34%` | narrow direct-copy NVTX around the source boundary |
| `residual coarse benchmark envelope: batch_prepare:decode:bs4` | `0.001838` | `0.25%` | narrow direct-copy NVTX around the source boundary |

## Notes

- Pre-instrumentation control built from existing sub-boundary summary because raw sqlite is not present in this workspace.
- Coarse batch_forward owners are intentionally residual for the 07.65 owner gate.

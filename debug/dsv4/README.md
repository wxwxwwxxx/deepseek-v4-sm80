# DeepSeek V4 developer benchmarks

The scripts under `debug/dsv4/benchmark/offline/` are development tools, not
stable public benchmark APIs. They cover DeepSeek V4 correctness probes,
microbenchmarks, profiling workloads, CUDA-graph lifecycle checks, and the
performance-matrix harness used while qualifying the release. Many assume a
DGX A100-SXM4-80GB system, tensor parallelism 8, and the model at
`/models/DeepSeek-V4-Flash`.

The supported public benchmark entry points are:

- `benchmark/offline/bench.py`
- `benchmark/offline/bench_wildchat.py`
- `benchmark/online/bench_simple.py`
- `benchmark/online/bench_qwen.py` (Qwen-format request trace replay)

Run developer scripts from the repository root. Archived TARGET reports and
older performance evidence may show their historical
`benchmark/offline/deepseek_v4_*.py` paths; those historical documents are
intentionally not rewritten. MTP is not part of this release; its history
remains in the archived prompts and the `dsv4-mtp-paused-reference` branch.

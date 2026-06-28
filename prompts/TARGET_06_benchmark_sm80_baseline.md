# TARGET 06: sm80 Baseline Benchmark and Bottleneck Map

## Goal

Build a repeatable A100/sm80 benchmark suite for the DSV4 mini-sglang implementation and use it to identify the next highest-value optimization work.

This target is complete when one command can generate baseline numbers for prefill, decode, memory, and fallback-kernel usage.

## Primary References

- Existing mini-sglang benchmark patterns:
  - `benchmark/offline/bench.py`
  - `benchmark/offline/bench_wildchat.py`
  - `python/minisgl/benchmark/perf.py`
- SGLang DSV4 performance behavior:
  - `/workspace/sglang-main/python/sglang/srt/models/deepseek_v4.py`
  - `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`

## Old dsv4 Branch References

These are especially useful for this target:

- `git show dsv4:benchmark/offline/deepseek_v4_bench.py`
- `git show dsv4:benchmark/offline/deepseek_v4_perf_matrix.py`
- `git show dsv4:benchmark/offline/deepseek_v4_perf_plot.py`
- `git show dsv4:benchmark/offline/deepseek_v4_runtime_env.py`
- `git show dsv4:prompts/PERF_RESULTS.md`

Reuse the good measurement ideas, but do not assume old branch performance conclusions still hold after switching to the sglang-based design.

## Plan

1. Add a DSV4 benchmark script.
   - Support local model path `/models/DeepSeek-V4-Flash`.
   - Support configurable prompt length, decode length, batch size, and number of repeats.
   - Record runtime environment: GPU, CUDA capability, torch version, optional kernel availability.

2. Measure core scenarios.
   - Single-request long prefill.
   - Batch prefill.
   - Decode throughput.
   - Mixed prefill/decode if scheduler supports it.
   - Repeated shared prompt with radix prefix disabled.

3. Track required metrics.
   - TTFT.
   - Prefill tokens/s.
   - Decode tokens/s.
   - End-to-end tokens/s.
   - Peak GPU memory.
   - KV cache memory.
   - Number of fallback wrapper calls.
   - Number of unsupported kernel skips.

4. Make output machine-readable.
   - Emit JSON or JSONL.
   - Include model path, git commit, branch, and config summary.
   - Include whether radix prefix is enabled; default should be false for this baseline.

5. Add bottleneck labeling.
   - Attribute obvious bottlenecks to:
     - attention
     - MoE / expert GEMM
     - fp4 expert handling
     - KV cache writes
     - metadata construction
     - scheduler overhead
   - Use wrapper counters from TARGET 05 when possible.

6. Add benchmark tests.
   - Smoke test with tiny prompt/decode sizes.
   - JSON schema/field test.
   - Runtime environment detection test.

## Done Criteria

- A repeatable command produces sm80 DSV4 benchmark results.
- Output includes timing, memory, environment, and fallback counters.
- First baseline can distinguish prefill bottlenecks from decode bottlenecks.
- Radix prefix remains disabled in the baseline unless explicitly requested.
- Old branch benchmark assets are referenced or reused only where they still fit.

## Non-Goals

- Performance tuning itself.
- Radix prefix cache performance measurement.
- Online serving benchmark.
- CUDA graph benchmarking unless already available.

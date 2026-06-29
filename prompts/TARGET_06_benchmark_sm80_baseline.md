# TARGET 06: sm80 Baseline Benchmark and Bottleneck Map

## Goal

Build a repeatable A100/sm80 benchmark suite for the DSV4 mini-sglang implementation and use it to identify the next highest-value optimization work.

This target is complete when one command can generate baseline numbers for prefill, decode, memory, and fallback-kernel usage.

All TARGET 06 performance numbers should use TP8 as the baseline configuration.
Do not spend TARGET 06 time comparing TP1/TP2/TP4 variants unless a later target
explicitly asks for tensor-parallel scaling. The goal here is a stable sm80
DSV4 baseline, not a TP scaling study.

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

## TARGET 05.7 Handoff

TARGET 05.7 proved minimum E2E viability, not performance:

- `/models/DeepSeek-V4-Flash` is a 149G checkpoint and OOMs on single A100-80GB
  TP1 during real-weight loading.
- The successful smoke used A100/sm80 with `torchrun --standalone` and
  PyTorch/NCCL collectives (`use_pynccl=false`).
- Fallback smoke passed with no DSV4 sm80 toggles active.
- `MINISGL_DSV4_SM80_V0_BF16=1` smoke passed and activated only the v0
  whitelist:
  - `MINISGL_DSV4_SM80_SWIGLU`
  - `MINISGL_DSV4_SM80_ROPE`
  - `MINISGL_DSV4_SM80_Q_NORM_ROPE`
  - `MINISGL_DSV4_SM80_KV_BF16`
  - `MINISGL_DSV4_SM80_COMPRESS`
  - `MINISGL_DSV4_SM80_COMPRESS_STORE`
  - `MINISGL_DSV4_SM80_TOPK`
  - `MINISGL_DSV4_SM80_INDEXER_BF16`
  - `MINISGL_DSV4_SM80_PAGED_MQA_BF16`
  - `MINISGL_DSV4_SM80_SPARSE_ATTN_BF16`
- Excluded experimental paths stayed disabled under the v0 bundle:
  `STORE_CACHE`, `FP4_GEMM`, `FP8_GEMM`, `WO_A_BF16`, `MOE_ROUTE`,
  `LINEAR_BF16_FP32`, fp8/fp4 indexer quantization, fp8/fp4 SwiGLU post-quant,
  and fp8 KV cache pack/dequant paths.
- Smoke artifacts:
  - `/tmp/dsv4_v0_fallback_smoke.json`
  - `/tmp/dsv4_v0_bf16_smoke.json`
  - corresponding `.rank0` through `.rank3` JSON files

Do not treat 05.7 elapsed seconds as performance data. Those smoke runs include
weight loading, initialization, and cold-start/JIT overhead. TARGET 06 must
separate model load/init time from measured prefill/decode time.

## Benchmark Policy

- Use TP8 for all official TARGET 06 numbers.
- Use `torchrun --standalone --nproc_per_node=8`.
- Use `distributed_init_method=env://` in the offline benchmark path when
  launched under `torchrun`.
- Use PyTorch/NCCL collectives for TARGET 06 (`use_pynccl=false`).
- Use `page_size=256` for all official performance benchmark numbers.
  `page_size=1` is allowed only for tiny smoke/debug paths and must not be
  reported as a performance baseline.
- Defer PyNCCL repair and PyNCCL performance comparison to a later target.
  PyNCCL currently reaches DSV4 TP E2E forward but fails during
  `lm_head.linear()` `all_gather` with `RuntimeError: unordered_map::at`.
- Primary variants:
  - `fallback`: all `MINISGL_DSV4_SM80_*` toggles cleared.
  - `v0_bf16`: only `MINISGL_DSV4_SM80_V0_BF16=1`.
- Additional experiment variants may be added after the baseline is stable, but
  they must remain separate from the primary v0 baseline:
  `MOE_ROUTE`, `WO_A_BF16`, `LINEAR_BF16_FP32`, fp8/fp4 GEMM, and other
  quantized activation/KV paths.

Suggested command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback v0_bf16 \
  --page-size 256 \
  --output-dir /tmp/dsv4_sm80_target06_tp8 \
  --keep-going
```

## Plan

1. Add a DSV4 benchmark script.
   - Support local model path `/models/DeepSeek-V4-Flash`.
   - Use TP8 as the only official baseline configuration.
   - Run under `torchrun --standalone --nproc_per_node=8`.
   - Use `distributed_init_method=env://` for torchrun-launched offline runs.
   - Default to PyTorch/NCCL collectives; do not enable PyNCCL in this target.
   - Default performance runs to `page_size=256`.
   - Keep `page_size=1` only for smoke tests that validate script wiring.
   - Support configurable prompt length, decode length, batch size, and number of repeats.
   - Record runtime environment: GPU, CUDA capability, torch version, optional kernel availability.
   - Record tensor parallel size, distributed init method, communication backend, and page size.
   - Split load/init timing from measured run timing.

2. Measure core scenarios.
   - Single-request long prefill.
   - Batch prefill.
   - Decode throughput.
   - Mixed prefill/decode if scheduler supports it.
   - Repeated shared prompt with radix prefix disabled.
   - Run both `fallback` and `v0_bf16` variants for each scenario.

3. Track required metrics.
   - Model load/init time, reported separately from benchmark timing.
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
   - Include TP size, rank count, distributed init method, communication backend, and page size.
   - Include active DSV4 sm80 toggles and raw DSV4 env values for every variant.
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

- A repeatable TP8 command produces sm80 DSV4 benchmark results.
- Output includes timing, memory, environment, and fallback counters.
- First baseline can distinguish prefill bottlenecks from decode bottlenecks.
- Baseline includes both `fallback` and `v0_bf16` variants.
- Reported timing separates load/init from measured prefill/decode windows.
- PyNCCL is explicitly disabled/deferred in TARGET 06 artifacts.
- Official performance artifacts use `page_size=256`.
- Any `page_size=1` result is labeled as smoke/debug and excluded from baseline summaries.
- Radix prefix remains disabled in the baseline unless explicitly requested.
- Old branch benchmark assets are referenced or reused only where they still fit.

## Non-Goals

- Performance tuning itself.
- Radix prefix cache performance measurement.
- Online serving benchmark.
- Tensor-parallel scaling study; TP8 is the only official TARGET 06 baseline.
- PyNCCL repair or PyNCCL-vs-NCCL benchmarking.
- CUDA graph benchmarking unless already available.

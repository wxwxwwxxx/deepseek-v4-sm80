# TARGET 05.7: DSV4 sm80 v0 bf16 Bundle and E2E Smoke

## Goal

Close the TARGET 05.x kernel adaptation work by creating a conservative
DeepSeek V4 sm80 v0 baseline switch and proving that the model can run a
minimal end-to-end generation smoke on A100/sm80.

This target is intentionally between TARGET 05.5 and TARGET 06:

- TARGET 05.5 remains the per-kernel R&D and microbenchmark record.
- TARGET 05.7 groups already validated low-risk bf16-direct kernels into one
  milestone toggle and checks E2E viability.
- TARGET 06 owns repeatable performance benchmarking, throughput reporting,
  and bottleneck analysis.

## Scope

Implement only the initial `bf16-direct` milestone path:

- no activation fp8/fp4 quantization;
- no fp8/fp4 KV cache storage;
- no radix prefix cache work;
- no broad performance matrix or bottleneck attribution.

The primary output of this target is a working opt-in policy plus E2E smoke
evidence, not a promoted default.

## v0 Bundle Policy

Add a single bundle environment switch:

```bash
MINISGL_DSV4_SM80_V0_BF16=1
```

When this switch is enabled on sm80, it should activate only the low-risk
bf16-direct kernels that already passed wrapper correctness and showed clear
microbench upside:

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

Keep the existing individual switches working for debugging and bisection.
The bundle switch should behave like an additional opt-in source, not a
replacement for individual toggles.

## Explicitly Excluded From v0 Bundle

Keep these paths behind separate switches because their E2E value, numeric
contract, or tuning status is not settled:

- `MINISGL_DSV4_SM80_STORE_CACHE`
- `MINISGL_DSV4_SM80_FP4_GEMM`
- `MINISGL_DSV4_SM80_FP8_GEMM`
- `MINISGL_DSV4_SM80_WO_A_BF16`
- `MINISGL_DSV4_SM80_MOE_ROUTE`
- `MINISGL_DSV4_SM80_LINEAR_BF16_FP32`
- fp8/fp4 indexer activation quantization
- fp8/fp4 SwiGLU post-quant kernels
- fp8 KV cache pack/dequant paths

These may become TARGET 06 or later experiment variants, but they should not
be part of the v0 baseline smoke.

## Implementation Plan

1. Add wrapper-level bundle policy.
   - Define `DSV4_SM80_V0_BF16_TOGGLE`.
   - Define a whitelist of per-kernel toggles included in the v0 bundle.
   - Update the DSV4 environment helper so a whitelisted toggle is considered
     enabled when either its individual env var or the bundle env var is true.
   - Keep sm80 and optional dependency checks unchanged.

2. Add tests for toggle semantics.
   - Verify `MINISGL_DSV4_SM80_V0_BF16=1` enables only the whitelist.
   - Verify excluded switches remain disabled unless explicitly set.
   - Verify individual switches still work without the bundle.

3. Add v0 wrapper smoke coverage.
   - Reuse the existing DSV4 wrapper parity tests.
   - Add a case that enables only `MINISGL_DSV4_SM80_V0_BF16`.
   - Confirm the same low-risk bf16 paths match their fallbacks.

4. Add minimal DSV4 E2E smoke.
   - Create a small offline smoke command or script for
     `/models/DeepSeek-V4-Flash`.
   - Run both variants:
     - fallback: all DSV4 sm80 env switches cleared;
     - v0 bf16: only `MINISGL_DSV4_SM80_V0_BF16=1`.
   - Use short prompt/decode settings so this target validates viability, not
     performance.

5. Record completion.
   - Update `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` with a short v0 bundle
     completion note and artifact paths.
   - Keep full performance numbers for TARGET 06.

## Suggested Smoke Commands

Correctness suite:

```bash
pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/models/test_deepseek_v4_forward_fallback.py
```

Fallback E2E smoke:

```bash
env -u MINISGL_DSV4_SM80_V0_BF16 \
  python -u debug/dsv4/benchmark/offline/deepseek_v4_e2e_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variant fallback \
  --prompt-len 16 \
  --decode-len 4 \
  --batch-size 1 \
  --output /tmp/dsv4_v0_fallback_smoke.json
```

v0 bf16 E2E smoke:

```bash
MINISGL_DSV4_SM80_V0_BF16=1 \
  python -u debug/dsv4/benchmark/offline/deepseek_v4_e2e_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variant v0_bf16 \
  --prompt-len 16 \
  --decode-len 4 \
  --batch-size 1 \
  --output /tmp/dsv4_v0_bf16_smoke.json
```

The smoke script should report pass/fail, generated token count, active DSV4
toggles, model path, GPU capability, torch version, and any exception message.
It should not attempt full throughput analysis.

For the full 149G `/models/DeepSeek-V4-Flash` checkpoint on A100-80GB, use
TP=4 instead of the single-process form above:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  debug/dsv4/benchmark/offline/deepseek_v4_e2e_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variant fallback \
  --prompt-len 16 \
  --decode-len 4 \
  --batch-size 1 \
  --output /tmp/dsv4_v0_fallback_smoke.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  debug/dsv4/benchmark/offline/deepseek_v4_e2e_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variant v0_bf16 \
  --prompt-len 16 \
  --decode-len 4 \
  --batch-size 1 \
  --output /tmp/dsv4_v0_bf16_smoke.json
```

## Done Criteria

- `MINISGL_DSV4_SM80_V0_BF16` exists and only enables the approved bf16-direct
  whitelist.
- All existing individual toggles remain available for bisection.
- Excluded experimental paths are not accidentally enabled by the bundle.
- Wrapper and forward fallback tests pass with the bundle enabled.
- `/models/DeepSeek-V4-Flash` completes minimal fallback E2E smoke.
- `/models/DeepSeek-V4-Flash` completes minimal v0 bf16 E2E smoke.
- TARGET 05.5 records the v0 smoke artifact paths.

## Handoff To TARGET 06

After this target passes, TARGET 06 should build the real benchmark matrix:

- prefill throughput;
- decode throughput;
- mixed workloads if scheduler support is ready;
- memory/KV cache reporting;
- fallback wrapper counters;
- bottleneck labels for attention, MoE, metadata, KV writes, and scheduler
  overhead.

TARGET 06 may use `MINISGL_DSV4_SM80_V0_BF16=1` as the primary sm80 baseline
variant and add separate experiment variants for `MOE_ROUTE`, `WO_A_BF16`,
`LINEAR_BF16_FP32`, and quantized GEMM paths.

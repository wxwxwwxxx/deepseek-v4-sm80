# TARGET 09.3: DSV4 SM80 FP8 KV/Cache Parity And Capacity Ledger

## Status

Ready after TARGET 09.0 if cache memory, cache bandwidth, or serving capacity is
a strong reason to investigate FP8 cache storage.

This target is source parity and accounting only.  Do not implement full FP8
KV/cache E2E in this target.

## Goal

Map SGLang/vLLM DeepSeek V4 FP8 KV/cache behavior and decide whether mini should
attempt a minimal FP8 cache slice.

The target must answer:

- Which DSV4 cache components can safely become FP8?
- Which components must stay BF16/FP32?
- Where do SGLang/vLLM quantize and dequantize?
- How many bytes/token would mini save?
- What correctness and graph-capture risks exist with radix prefix cache?

## References

Mini:

- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/engine/graph_runner.py`

SGLang:

- `/workspace/sglang-main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py`
- `/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/`

vLLM:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/`

## Required Work

1. Component map

   List current mini DSV4 cache components:

   - compressed/MLA KV state;
   - RoPE tail;
   - SWA state;
   - C4/C128/indexer state;
   - prefix-cache component ownership metadata;
   - scales/workspace.

2. SGLang/vLLM parity map

   For each component, report:

   - dtype;
   - layout and stride;
   - scale format;
   - quantize/store location;
   - gather/dequant location;
   - decode versus prefill behavior;
   - SM80 compatibility.

   Mark each conclusion as source-derived or runtime-proven.

3. Capacity ledger

   Compute:

   - bytes/token before and after;
   - GiB/rank saved for `--page-size 256 --num-pages 128`;
   - equivalent extra pages/tokens;
   - graph capture memory headroom impact;
   - prefix-cache retained-page impact;
   - extra scales/workspace.

4. Correctness risk map

   Identify risks for:

   - prefix-cache hits and remaps;
   - Route B component ownership;
   - SWA tail retention;
   - graph-captured metadata buffers;
   - logit drift and text-smoke drift.

5. Recommendation

   Decide whether TARGET 09.4 should run and which minimal slice it should use.
   Prefer the smallest slice that tests the real store/gather/dequant boundary.

## Gates

Pass if:

- SGLang/vLLM behavior is mapped clearly enough to avoid inventing a new layout;
- bytes/token and capacity savings are concrete;
- a minimal slice is proposed with exact boundaries;
- correctness risks have testable checks.

Stop if:

- source parity cannot identify an SM80-compatible FP8 cache path;
- saved memory is too small after scales/workspace;
- dequant requires full-cache dequantization;
- prefix-cache or graph constraints are incompatible with the proposed layout.

## Deliverables

Write results under:

```text
performance_milestones/target09_fp8_kv_cache_parity_ledger/
```

Include:

- `README.md` with TARGET 09.4 go/no-go;
- component dtype/layout table;
- capacity ledger;
- SGLang/vLLM parity notes;
- correctness risk table;
- proposed minimal slice boundaries.


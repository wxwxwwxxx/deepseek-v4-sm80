# TARGET 04: DSV4 Attention Backend and Metadata

## Goal

Implement the DSV4-specific attention backend and metadata construction needed for SWA, compressed C4/C128 attention, and indexer-driven sparse selection.

This target is complete when DSV4 prefill and decode can build correct metadata and call a fallback attention implementation through the same API that future fused kernels will use.

## Primary References

- Main backend reference: `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- Metadata reference: `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py`
- Compressor reference: `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/compressor.py`
- Indexer reference: `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/indexer.py`
- Current local attention code:
  - `python/minisgl/attention/base.py`
  - `python/minisgl/attention/fa.py`
  - `python/minisgl/attention/fi.py`
  - `python/minisgl/attention/trtllm.py`

## Old dsv4 Branch References

Useful for scheduler/metadata experiments:

- `git show dsv4:python/minisgl/attention/deepseek_tilelang.py`
- `git show dsv4:python/minisgl/kernel/deepseek_v4.py`
- `git show dsv4:tests/kernel/test_deepseek_tilelang.py`

Use old TileLang attention as a behavioral reference only. The new backend should expose sglang-style DSV4 metadata.

## Plan

1. Add a DSV4 attention backend class.
   - Do not force the API into ordinary MHA if required metadata is missing.
   - Keep one public `forward` entry point for model code.
   - Internally separate prefill, decode, and fallback paths.

2. Add DSV4 attention metadata.
   - Include:
     - raw output locations
     - page table
     - sequence lengths
     - positions
     - SWA page indices
     - SWA top-k lengths
     - C4 compressed locations
     - C128 compressed locations
     - C4 indexer locations
     - sparse raw indices or page indices
   - Keep shapes close to sglang-main to simplify later kernel porting.

3. Build metadata from scheduler state.
   - Use the DSV4 KV pool from TARGET 03.
   - Respect `window_size=128` for SWA.
   - Respect `compress_ratios` for ratio-specific layers.
   - Decode metadata must refer to full logical context even when physical cache components differ.

4. Implement fallback attention.
   - Use torch fallback or a simple existing local attention backend.
   - Preserve output shape expected by DSV4 `MQALayer`.
   - Make sparse/compressed paths correctness-first.

5. Add compressor/indexer hooks.
   - Wire placeholders for compressor output and C4 indexer top-k.
   - If high-performance top-k transform is not available, use a deterministic fallback.

6. Add metadata tests.
   - Sequence length and position construction.
   - SWA window boundary at lengths below, equal to, and above 128.
   - Ratio `0`, `4`, and `128` layer dispatch.
   - Multi-request batch page index construction.

## Done Criteria

- DSV4 model code can call `DSV4AttentionBackend`.
- Metadata is constructed for prefill and decode.
- Fallback attention returns correctly shaped output.
- Tests cover SWA windowing, compress ratio routing, and batched metadata.
- Fused kernel integration points are explicit and stable.

## Non-Goals

- Final FlashMLA performance.
- Radix prefix cache.
- CUDA graph capture.
- sm100 fp4 indexer path.

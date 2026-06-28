# P0 HIGH: paged_mqa_attention_fallback bf16 Plan

## Why This Is High Priority

The attention fallback is structurally expensive because it uses Python ragged
indices and small torch operations per query. Clearing it is required before
the DSV4 attention path can be considered mostly fallback-free.

## Policy

- Keep q and KV activations bf16 first.
- Do not introduce fp8 KV cache for the first replacement.
- Prefer upstream FlashInfer or FlashMLA if the required cache layout can be
  adapted cheaply on sm80. If not, implement a flat-cache bf16 kernel.

## Current State

Mini has bf16 flat cache buffers and per-row context index tensors. FlashInfer
MLA wrappers exist in the environment, but they require paged ckv/kpe cache
layout and packed metadata.

## Typical Workloads

- Decode: active tokens `1..batch`, local heads, `head_dim=128`.
- SWA-only candidate count bounded by `window_size`.
- C4/C128 layers combine compressed candidates and SWA candidates.

## Implementation Plan

1. Normalize metadata first.
   - Replace Python list input with `indptr`, `indices`, and row lengths.
   - Preserve duplicate removal and ordering exactly.

2. Try upstream backends with an adapter.
   - Build a FlashInfer/FlashMLA metadata adapter prototype.
   - Include adapter allocation/copy cost in the microbench.
   - Keep it only if it is cheaper than a local kernel.

3. Implement local bf16 flat-cache decode if upstream layout is not viable.
   - One program per query/head or query/head block.
   - Load candidate rows by flat cache index.
   - Compute scores in fp32, apply attention sink, reduce values to bf16.

4. Add fp8 KV dequant only after the bf16 path is stable and useful.

## Validation

- Parity against `paged_mqa_attention_fallback` for empty, single, SWA-only,
  C4+SWA, C128+SWA, and duplicate candidate rows.
- Microbench candidate counts 32, 128, 640, 1024.
- No E2E requirement until other major fallbacks are removed.

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `paged_mqa_attention_fallback` with correctness, microbench, decision, and
artifact paths.

## 实现结论

哈哈，是的，这个算子的结论可以说是**相当乐观，但还没到“默认打开”的程度**。

当前状态：

- `paged_mqa_attention_fallback` 已经从“研究未实现”推进到 **implemented opt-in**。
- 新路径是 `bf16-direct flat cache Triton decode`。
- 开关是：`MINISGL_DSV4_SM80_PAGED_MQA_BF16=1`。
- 原 torch fallback 完整保留，不满足条件会自动回退。
- 已新增结构化 metadata：`DSV4PagedMQAMetadata(indptr, indices, lengths, max_length)`，解决了原来 Python ragged list 作为 kernel 边界的问题。
- backend 现在会先规整 metadata，再交给 attention wrapper。

验证结果挺漂亮：

- CUDA parity 通过，覆盖 empty / single / duplicate rows / sink / no-sink。
- 默认 smoke：`20 passed`
- 开启新 kernel smoke：`20 passed`
- microbench 候选数 `32/128/640/1024` 全部 allclose 通过。
- 性能大概是：
  - fallback 约 `3.3 ms`
  - Triton metadata 路径约 `0.058 - 0.089 ms`
  - speedup 约 `37.85x - 57.19x`

所以直觉上，这个算子是目前这批里非常好的一个：它正好打中了原 fallback 的痛点，也就是 Python per-row 小 torch op 循环。

但我会保守一点说：**乐观，但还不是默认 promotion 结论**。原因是：

- 当前性能证据是 microbench，还没有完整 DSV4 E2E perf matrix。
- 现在主要覆盖 decode 风格、bf16 flat cache、最多 1024 candidates。
- list 到 metadata 的构造成本如果在真实路径里频繁发生，也要看 E2E 里是否被摊薄或复用。
- 还没做 fp8 KV dequant，也没接 FlashMLA/FlashInfer paged layout。

一句话总结：  
**这个 P0 算子已经从“结构性 fallback 瓶颈”变成了“很有希望的 opt-in 加速路径”，数据非常正面，下一步应该用真实 DSV4 decode/E2E 工作负载确认是否能推广。**
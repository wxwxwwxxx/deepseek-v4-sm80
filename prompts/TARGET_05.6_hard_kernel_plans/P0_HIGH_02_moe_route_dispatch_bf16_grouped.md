# P0 HIGH: moe_gate/hash_topk/mega_moe_pre_dispatch bf16 Grouped Plan

## Why This Is High Priority

MoE routing plus expert dispatch is on every routed layer. Even a good expert
GEMM cannot help if tokens still flow through Python expert loops and many
small fallback calls.

## Policy

- Keep activations bf16 through routing and dispatch.
- If expert weights are fp4/fp8, dequant or upcast weights inside the grouped
  GEMM consumer, then use bf16 tensor cores.
- Prefer upstream route kernels first. Use SGLang JIT route helpers only if they
  compile and run on sm80. Avoid dense invalid-route graph fallbacks as default.

## Current State

`moe_gate_fallback` uses torch scoring/top-k, and routed experts are selected
with Python loops over expert ids. Upstream has `hash_topk`,
`mask_topk_ids`, and `mega_moe_pre_dispatch`, but local sgl-kernel DSV4 ops are
not exposed.

## Typical Workloads

- Decode: small token count, `topk` routes per token.
- Prefill: larger token count, more route reuse.
- Hash layers use token-id table lookup; non-hash layers use router scores.

## Implementation Plan

1. Split the work into route metadata and expert compute.
   - Route metadata: sorted token ids, expert ids, per-expert counts, offsets,
     routed weights.
   - Expert compute: grouped dequant-on-load bf16 tensor core GEMM.

2. Try upstream route helpers first.
   - Attempt SGLang JIT route kernels in isolation.
   - Keep them only if they are sm80-compatible and do exact route grouping.

3. Implement exact local route grouping if upstream is blocked.
   - No dense invalid-route computation.
   - No Python loop over experts in the hot path.
   - Preserve deterministic accumulation order.

4. Connect to `quantized_linear_ref` replacement.
   - Dispatch grouped hidden states into expert chunks.
   - Reuse the P0 quantized-weight dequant-on-load bf16 tensor core plan.

## Validation

- Route metadata parity with current torch fallback.
- Full routed expert output parity.
- Microbench routing alone and route+GEMM together.
- 基础 route-dispatch 阶段不强制 E2E；V1_MOE promotion 的 E2E gate 和长跑证据
  在下方单独记录。

## Matrix Update Requirement

After implementation or a serious failed attempt, update
`prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` in the R&D Completion Matrix row
for `moe_gate/hash_topk/mega_moe_pre_dispatch` with correctness, microbench,
decision, and artifact paths.

## 基础实现结论

本文件最早跟踪的是 `moe_gate/hash_topk/mega_moe_pre_dispatch` 这一组
route-dispatch + grouped expert compute 能力。基础能力已经落地：

- 本地 `DSV4MoERoutePlan` 可以把 `[tokens, topk]` route 精确展开、按 expert
  排序并按 block padding。
- Triton grouped FP4 linear kernel 可以在 grouped MoE 中对 w1/w3/w2 packed
  FP4 权重做 dequant-on-load，再走 bf16 Tensor Core dot。
- `DSV4FusedRoutedExperts.forward` 在
  `MINISGL_DSV4_SM80_MOE_ROUTE=1` 且 sm80/Triton 可用时尝试 grouped 路径；
  否则保持原逐 expert fallback。

v0 bundle 中该路径仍保持 opt-in；`MINISGL_DSV4_SM80_V1_MOE=1` 已经把这些
能力组合成 V1 exact grouped MoE 默认候选路径。V1 的完整实现状态、E2E
证据和下一步优化空间见下方 `V1_MOE` 小节。

## TARGET 06 后续: MoE v1/v2 硬算子计划

这仍然属于算子级优化。主要性能面在 routed expert 的 dispatch、FP4
weight dequant、grouped GEMM、SwiGLU、routed-weight combine 和 TP reduce
边界；需要少量模型侧 wiring 来减少 all-reduce 次数，但目标仍是把多个
fallback wrapper 收敛成可验证的 MoE operator，而不是改变 serving scheduler
或高层模型语义。

### TARGET 06 性能证据

早期 `sm80_v0_bf16` E2E baseline 显示 prefill 已经大幅改善，但 decode 仍被
MoE/FP4 expert 路径卡住：

- `quantized_linear_ref`: wrapper call time 约 40%。
- `dequant_fp4_weight`: wrapper call time 约 37%-38%。
- `silu_and_mul_clamp_fallback`: wrapper call time 约 13%。
- `moe_route_dispatch_bf16_grouped`: v0 bundle 尚未启用，仍无法作为默认 E2E
  路径消除逐 expert fallback/ref compute。

A100/sm80 没有原生 FP4/FP8 Tensor Core。默认高精度路线应继续是
FP4 weight dequant-on-load + bf16 Tensor Core MMA；不要把 DeepGEMM 或
FlashInfer 的 sm90/sm100 MoE 路线作为 sm80 默认依赖。

### 当前路径

当前 routed MoE 热路径可以简化为：

1. `moe_gate_fallback` 用 torch 计算 router score/top-k。
2. `DSV4FusedRoutedExperts.forward` 尝试 `moe_route_dispatch_bf16_grouped`。
   在 v0 bundle 中该开关未启用，或 grouped path 返回 `None` 时，会落回逐
   expert Python loop。
3. 每个 expert 分别执行 `w1`、`w3`、SwiGLU、`w2`。
4. `w1/w3/w2` 通过 `quantized_linear_ref` 反复 materialize FP4 dequant
   weight，再调用 torch linear。
5. routed expert 和 shared expert 的 TP partial 结果在模型侧规约。

这个路径的问题不是单个 attention kernel，而是 MoE expert compute 被切成了
大量小 wrapper 调用，并且 FP4 权重反量化没有和 GEMM 消费融合。

### 理想路径

理想的 exact MoE 路径是：

1. gate/top-k 产生精确 route metadata。
2. route metadata 把 token 按 expert 分组，生成 contiguous/grouped expert
   chunks。
3. grouped FP4 GEMM 在读 weight tile 时完成 nibble decode、scale apply 和
   bf16 cast，不生成完整 dequant weight。
4. `w1`/`w3` 尽量在同一 grouped expert pipeline 中产生 gate/up activation。
5. SwiGLU、clamp、routed weight 乘法尽量和后续 `w2` 消费贴近，减少中间
   activation materialization。
6. routed expert local partial 与 shared expert local partial 先在 rank 内相加，
   再做一次 TP all-reduce。

### V1_MOE: exact grouped MoE default path

`V1_MOE` 是当前 V1 的默认候选路线。它只做 exact/bf16-direct 优化，不引入
INT8 approximation；目标是在保持模型语义的前提下，把 routed FP4 expert 从
Python expert loop、完整 FP4 dequant materialization 和反复
`quantized_linear_ref` 中移出。

默认开关和语义：

- `MINISGL_DSV4_SM80_V1_MOE=1` 等价于 v0 bf16 bundle 加
  `MINISGL_DSV4_SM80_MOE_ROUTE=1`；不会启用 V2 INT8 approximate path。
- Attention 之后的 hidden states 直接以 bf16 activation 进入 grouped MoE；
  V1 不再做 activation quant 来复刻旧 `quantized_linear_ref` 语义。
- FP4 expert weight 在 grouped GEMM consumer 内 dequant-on-load，然后用 bf16
  Tensor Core dot；不生成完整 dequant weight。
- `DSV4MoE.forward` 在 V1 下让 routed expert 和 shared expert 先返回
  rank-local partial，本地相加后只做一次 TP all-reduce。

五个开发阶段完成状态：

1. 已新增 `MINISGL_DSV4_SM80_V1_MOE=1` bundle，并把 perf matrix、text smoke
   和 E2E smoke 的候选 variant 扩展为 `v1_moe`。
2. `build_moe_route_plan` 在 sm80/Triton/MoE toggle 下优先走本地 Triton route
   metadata kernels，替换 torch `argsort/bincount/padding` 热路径；CPU、非
   CUDA 和 kernel 失败时保留 torch fallback。
3. grouped MoE 的 `w1`/`w3` 已合并为一次 grouped FP4 producer，同一个 route
   tile 同时产生 gate/up activation，避免两次 dispatch 和重复 hidden-state
   读取。
4. V1 先尝试 guarded single-kernel fused compute:
   `route-grouped fused compute -> per-route output -> deterministic route sum`。
   该 kernel 在 activation 前显式模拟 w1/w3 bf16 materialization 边界，保持
   bf16-direct oracle 语义。若 shape/perf guard 不满足，则回落到 V1 pipeline:
   `grouped w1w3 producer -> fused SwiGLU/clamp/routed-weight activation ->
   grouped w2 consumer -> deterministic route sum`。
5. routed expert 和 shared expert 的 reduce 边界已下移到 rank-local sum 之后；
   旧的 routed/shared 各自 reduce 路径保留为 fallback，并有等价性测试覆盖。

当前实现路径：

1. `moe_gate_fallback` 继续产生 exact route weights/indices。
2. `build_moe_route_plan` 生成 sorted route ids、expert ids、padded block
   metadata 和 route weights。
3. `moe_route_dispatch_bf16_grouped` 进入 grouped MoE。
4. 小/中 hidden shape 先尝试 guarded single-kernel fused compute。
5. 真实 DSV4 hidden=4096 等大 hidden shape 命中 guard，自动使用 V1 pipeline，
   避免 single-kernel 按 output hidden tile 重算 w1/w3。
6. deterministic route sum 将 per-route output 还原为 token output。
7. routed local partial 与 shared local partial 先本地相加，再做一次 TP
   all-reduce。

验证与证据：

- 单测:
  `pytest -q -o addopts='' tests/kernel/test_deepseek_v4_wrappers.py
  tests/models/test_deepseek_v4_forward_fallback.py
  tests/benchmark/test_deepseek_v4_perf_matrix.py
  tests/benchmark/test_deepseek_v4_text_smoke.py` 通过，39 passed。
- Microbench:
  `/tmp/dsv4_moe_route_dispatch_bf16_grouped_v1_fused_compute_microbench.json`。
  full grouped path 相对当前 fallback: `decode_tiny` 19.65x、
  `decode_grouped` 85.96x、`prefill_grouped` 58.40x；route metadata kernel
  约 0.14 ms；grouped vs bf16 oracle max_abs 为 0.0/0.0/0.25。
- TP=8 smoke:
  `/tmp/dsv4_sm80_v1_moe_fused_compute_smoke/summary.json`，`v1_moe` passed；
  elapsed 5.94s，decode tok/s 2.80，fallback wrapper calls 44672，
  `dequant_fp4_weight` calls 0，`moe_route_dispatch_bf16_grouped` 无
  optional-none skips。
- E2E gate:
  `/tmp/dsv4_v1_moe_e2e_gate`。`decode_throughput_bs8` 中 v0 -> V1 decode
  tok/s 2.88 -> 22.53，提升 7.83x；elapsed 565.15s -> 73.32s；fallback
  wrapper calls 16.72M -> 2.14M。`mixed_prefill_decode_bs4` 中 v0 -> V1 decode
  tok/s 1.49 -> 7.20，提升 4.82x；elapsed 195.12s -> 43.38s；fallback
  wrapper calls 5.03M -> 1.06M。
- 4096/1024/bs4 长跑:
  `/tmp/dsv4_v1_moe_4096x1024_bs4`。`v0_bf16` -> `v1_moe`: elapsed
  2188.35s -> 389.80s，提升 5.61x；TTFT 27.21s -> 24.26s，提升 1.12x；
  prefill tok/s 621.71 -> 695.41，提升 1.12x；decode tok/s 1.90 -> 11.25，
  提升 5.94x；E2E output tok/s 1.87 -> 10.51，提升 5.61x；fallback wrapper
  calls 56.64M -> 11.44M。

V1_MOE 结论：

- V1_MOE 可以作为 V1 variant 的默认路径。它已经在 microbench、TP=8 smoke、
  E2E gate 和 4096/1024/bs4 长跑中给出稳定收益。
- routed FP4 expert 的主要灾区已经解决：V1 中
  `moe_route_dispatch_bf16_grouped` 无 optional-none skips，`dequant_fp4_weight`
  calls 归零，Python expert-loop/ref FP4 compute 大幅下降。
- 4096/1024/bs4 显示 V1 的主要收益来自 decode，decode tok/s 提升 5.94x；
  TTFT/prefill 只提升约 1.12x，说明长 prefill 的瓶颈已经转移到 routed MoE
  之外。
- 当前 guarded single-kernel fused compute 不应直接扩展到真实 DSV4 hidden=4096。
  该版本按 output hidden tile 切分时会重算 w1/w3；直接放宽 guard 可能反向拖慢。
- V1_MOE 是相对 v0 的 promotion candidate，但还不是最终性能形态。4096/1024/bs4
  的 E2E output tok/s 为 10.51，约为外部 114 tok/s 目标的 9.2%。

下一阶段优化空间按优先级排列：

1. 优先优化 V1 后的新主瓶颈: `dequant_fp8_weight`、`quantized_linear_ref`、
   shared/dense FP8 linear、`linear_bf16_fp32_fallback`。在 4096/1024/bs4 中，
   V1 残留 top calls 包括 `dequant_fp8_weight` 2.64M、
   `quantized_linear_ref` 2.29M、`linear_bf16_fp32_fallback` 0.71M。
2. 做真实 DSV4 大 hidden 友好的 MoE V1.1，而不是放宽当前 single-kernel guard:
   `kernel A: grouped w1/w3 + SwiGLU/clamp/routed-weight -> activated per-route bf16`;
   `kernel B: grouped w2 consumer`; `kernel C: deterministic route sum`。如果
   kernel B 能安全合并 route sum，再作为后续阶段推进。
3. 优化 HC/MLA/attention 残余路径，包括 `hc_pre_fallback`、`hc_post_fallback`、
   `compress_forward_fallback`、`apply_rotary_tail`、`q_norm_rope_fallback`、
   `k_norm_rope_cache_fallback` 和 `dsv4_sparse_attention_two_source_bf16`。
4. 继续降低 route metadata 固定成本，尤其是 decode 小 batch 下的 route count/
   offset/fill kernel overhead。该项重要，但优先级低于当前更大的 FP8/dense
   residual hotspots。
5. V2 INT8 仍保持独立实验路线；只有当 V1.1 exact path 和 shared/FP8 残余路径
   稳定后，再评估 approximate INT8 是否值得引入。

阶段性 promotion gate:

- `decode_throughput_bs8` decode tok/s 至少达到当前 v0 的 5x: 已完成。
- `mixed_prefill_decode_bs4` 不低于当前 v0，且无 TTFT 明显回退: 已完成。
- 4096 input / 1024 output / batch 4 可比 benchmark 必须能生成报告: 已完成，
  但绝对 throughput 距离 114 tok/s 目标仍有明显差距。

### FP4 dequant policy

FP4 expert exact 路线必须避免复杂数学函数：

- Python fallback 继续使用 `_fp4_table` LUT 作为 correctness oracle。
- Triton/CUDA kernel 中使用 nibble unpack + branchless/table-like constants；
  不在 FP4 value decode 中使用 `exp`/`exp2`。
- E8M0 scale decode 可以单独优化为 cached bf16/fp32 scale shadow，前提是
  内存增长可控，并且 checkpoint/load path 有明确 invalidation。
- scale 和 packed FP4 weight 的 layout 优先服务 grouped MoE 的 tile 访问，
  不为单次 `quantized_linear_ref` 小收益牺牲主路径。

### V2 INT8 Tensor Core opt-in

`V2` 是高风险 approximate 路线，必须独立开关，不进入默认 bundle：

- 开关建议: `MINISGL_DSV4_SM80_MOE_INT8=1`，且要求
  `MINISGL_DSV4_SM80_MOE_ROUTE=1`。
- load/checkpoint 后为 routed expert FP4 weight 构建 INT8 shadow weight 和
  scale metadata，避免每 token 动态重编码。
- activation 使用 per-token 或 per-block INT8 quant，走 sm80 INT8 Tensor Core
  grouped GEMM。
- 输出恢复到 bf16，并复用 grouped route/reduce pipeline。
- 该路线只在 V1 exact grouped MoE 稳定后实现；不能用它掩盖 exact path 的
  routing/fusion 问题。

V2 correctness gate 需要比 V1 更严格：

- 单层 routed expert 对 bf16-direct oracle 的 max/mean error、cosine similarity。
- 多层 forward smoke 与文字 smoke，无乱码、无明显语义崩坏。
- 固定 prompt 的 fallback/v1/v2 token-level 对比报告。
- 如果 INT8 路线不能比 V1 exact grouped MoE 再快至少 1.5x，则保留为实验
  路径，不继续推广。

### 测试与 benchmark 口径

新增或复用以下验证层级：

- route metadata parity: token ids、expert ids、offsets、weights 精确匹配。
- grouped expert parity: 小 shape exact parity，大 shape 对 bf16-direct oracle
  设定明确 tolerance。
- reduce equivalence: 旧的 routed/shared reduce 顺序与新的一次 all-reduce
  输出一致或在 tolerance 内。
- text smoke: page size 固定 256，TP=8，fallback/v0/v1/v2 分别跑简单中英文
  prompt。
- microbench: route-only、w1w3 grouped、activation、w2 grouped、full MoE
  pipeline 分段报告。
- E2E: TARGET 06 matrix 加 `v1_moe`；另加 4096 input / 1024 output / batch 4
  可比 benchmark。

### 主要风险

- route imbalance 会让 grouped GEMM 变成许多小 expert chunks，需要按 decode
  和 prefill 分别调 block size。
- 过度融合可能增加 register pressure，导致 occupancy 下降；实现时以 E2E
  decode gain 为准，不以 kernel 数量最少为唯一目标。
- INT8 shadow weight 会增加显存占用，必须在 TP=8 A100-80GB 的真实 load
  path 验证。
- approximate INT8 可能影响 MoE routing 后的语义稳定性，必须保持 opt-in。

你好，请帮我在这个项目中调研一下应该如何将Deepseek-V4-Flash植入mini-sglang框架，并完成模型在sm80下的适配。相关信息如下：
1、需要植入的框架是/workspace/mini-sglang，这是一个简单的大模型推理框架，其中实现了sglang的关键功能；
2、需要植入的模型的位置是：/models/DeepSeek-V4-Flash；
3、模型推理相关的代码参见/models/DeepSeek-V4-Flash/inference文件夹（官方oracle）和/workspace/sglang-main（sglang仓库）；
4、要点：
    a) 我们的目标是在当前的mini-sglang中实现deepseek v4的高性能推理，并完成模型在sm80下的适配工作；
    b) 我们之前在当前仓库的dsv4分支中做过一些尝试，但是我分析后发现性能较差且植入的工作量太高，因此我放弃了基于原始模型代码的这个方案，并重建了一个新分支，名为dsv4-sglang-based，也就是当前分支；然而，dsv4分支中仍然可能有一些值得参考的内容，比如模型的oracle版本，请按需使用。
    c) 为了解决上面的问题，我决定不应该直接参考原始模型代码，而是应该参考sglang中已有的高性能实现，并找到sm80不支持选项的操作（比如DeepGEMM等算子），针对性的完成适配。为此，你可以参考sglang代码仓库，本机中的代码位置在/workspace/sglang-main。由于它代码量庞大且依赖了复杂的软件包，因此我没有安装他，辛苦你直接阅读他，并基于他的实现，找到DeepSeek V4相关的设计并移植到本代码仓库。
5、实现路径：
    本仓库里存在基础的radix前缀缓存实现。我设想的实现路径如下：
    a) 完成radix tree到DSV4 kvcache的适配。DSV4模型中涉及到多种不同的KV cache，如C4A attention、C4A indexer和C128A attention，以及SWA。请参考sglang中的设计，将其移植到本仓库。注意在本机sm80的环境下，我们存在两种选择，一是在算子中做低精度适配，并保存在低精度的kvcache中。二是算子中不做低精度适配，直接保存bf16 kvcache。代码中可以留下相应接口；适配可以分两步，第一步可以暂时不支持前缀缓存，待初步性能确认后再来实现SWA相关的前缀缓存。
    b) 找到sglang中的算子融合方案。并将其移植到本机。第一步我们可以找到sglang在sm90、sm100下如何设计sglang的计算图，调用了哪些大融合算子。你可以将其接口写出来，并标记todo。我会去寻找相关算子的实现，后续我们一起将其移植到sm80。sglang仓库中应当有这些算子的接口，不过如果他们依赖了第三方仓库，本机中可能没有安装，你可以列出来后，我去github下载它的最新代码。
6、如果存在我没有提到的难点，你可以列出来并单独详细说明！

## 阶段 Matrix

| Stage | Prompt | Status | Completion Record |
| --- | --- | --- | --- |
| TARGET 05.5 | `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` | planned | DSV4 sm80 高性能算子替换研发计划已创建。后续每次替代 kernel 时，需要在该文件的 R&D Completion Matrix 中记录 kernel、mode、toggle、correctness、microbench、E2E perf、decision 和 artifact 路径。 |
| TARGET 05.7 | `prompts/TARGET_05.7_dsv4_v0_bf16_e2e_smoke.md` | completed | 已新增 `MINISGL_DSV4_SM80_V0_BF16` 白名单 bundle、语义测试、wrapper bundle smoke、最小离线 E2E smoke 脚本，并验证 `/models/DeepSeek-V4-Flash` fallback/v0_bf16 在 A100 sm80 TP=4 下均生成 4 tokens。Artifact: `/tmp/dsv4_v0_fallback_smoke.json`, `/tmp/dsv4_v0_bf16_smoke.json`。正式性能矩阵交给 TARGET 06。 |
| TARGET 06 | `prompts/TARGET_06_benchmark_sm80_baseline.md` | completed | 已新增 torchrun-native TP8 benchmark harness `benchmark/offline/deepseek_v4_perf_matrix.py`，默认 page_size=256、PyTorch/NCCL、radix disabled，覆盖 fallback/v0_bf16、prefill/decode/shared-prefix 场景，输出 JSON/JSONL、环境、内存、fallback counters 和瓶颈标签；修复 Engine page table 对 page_size=256 尾页写入的对齐问题；新增正式评测前的文本正确性 smoke `benchmark/offline/deepseek_v4_text_smoke.py`，同样默认 TP8/page_size=256，并记录回复文本、解析结果和乱码/复读/期望答案检查。本轮修复了 TP8 正确性问题：DSV4 TP routed experts 缺失 all-reduce、`attn_sink` 权重未按 local heads 分片、fallback q_norm_rope 未原地写回 query，并补齐 fallback two-source sparse attention 读取压缩 cache。验证：纯逻辑/schema 测试通过，TP8 page_size=256 tiny smoke 通过 fallback/v0_bf16，artifact: `/tmp/dsv4_target06_smoke_variants/summary.json`；decode phase smoke artifact: `/tmp/dsv4_target06_smoke_decode/summary.json`；text smoke artifacts: `/tmp/dsv4_text_smoke_after_qnorm_fix.json`、`/tmp/dsv4_text_smoke_full_after_qnorm_fix.json`，fallback/v0_bf16 在 3 条简单提示上均为 `pass`。正式长 baseline 可用 TARGET 06 suggested command 直接生成，smoke/debug 结果不计入官方 baseline。 |
| TARGET 07 | `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` | planned | DSV4 sm80 vLLM gap closure 新主路线图。当前 best exact 为 Marlin WNA16 + global topk/lens + bf16 split-K sparse decode，4096/1024/batch4 达到 `68.81 output tok/s`；当前 opt-in FP8 indexer + FP8 activation quant Triton 路径达到 `87.08 output tok/s`。07.56 的 static scale cache 只带来 `+0.35%`，低成本 graph/layout preflight 已结束；下一步运行 TARGET 07.57，正式对 projection/GEMM backend 与 vLLM 做 owner-level parity 和 PoC。 |
| TARGET 07.10 | `prompts/TARGET_07.10_dsv4_sm80_foundation_history.md` | completed history | 合并 07.1/07.2/07.25：公平复测、通信/CUDA graph、subgraph parity。旧细粒度 prompt 保留为 archive。 |
| TARGET 07.20 | `prompts/TARGET_07.20_dsv4_sm80_moe_history.md` | completed history | 合并 07.3/07.35/07.36/07.37/07.38/07.39/07.391：MoE 从 V2 到 mini-owned Marlin WNA16 的完整结论；MoE 目前不再是 primary bottleneck。 |
| TARGET 07.30 | `prompts/TARGET_07.30_dsv4_sm80_attention_history.md` | completed history | 合并 07.392/07.393/07.394/07.395：attention/indexer/cache 从 post-Marlin profile 到 global topk/lens，再到 exact bf16 split-K sparse decode；decode sparse boundary 已基本打平 vLLM 对应 probe。 |
| TARGET 07.40 | `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md` | completed | post-splitK reprofile：重新区分 decode split-K、legacy prefill sparse、indexer/cache、runtime/copy；结论是 decode split-K 已不是主瓶颈，下一步进入 exact runtime/indexer/cache。 |
| TARGET 07.41 | `prompts/TARGET_07.41_dsv4_sm80_indexer_cache_runtime_exact.md` | completed | exact replay metadata-copy cut：microbench 明显变好，但 4096/128 和 4096/1024 macro 基本不变；结论是不要继续局部 metacopy polish，需要重新对齐 vLLM 证据链。 |
| TARGET 07.42 | `prompts/TARGET_07.42_dsv4_sm80_vllm_metadata_runtime_parity.md` | completed | evidence-first mini-vs-vLLM metadata/runtime/indexer/cache parity：未找到足够支撑 exact runtime PoC 的证据；推荐 precision/cache，但 vLLM per-bucket timing 仍不完整。 |
| TARGET 07.43 | `prompts/TARGET_07.43_dsv4_sm80_vllm_ablation_before_precision.md` | completed | vLLM 破坏性实验：aux stream off 仅 `-0.54%`，persistent topk off 为 `+0.21%`，eager 大幅掉速但 mini 已有 decode graph；结论是直接进入 vLLM-aligned FP8 cache/indexer lane。 |
| TARGET 07.50 | `prompts/TARGET_07.50_dsv4_sm80_fp8_cache_indexer_precision.md` | completed | vLLM-aligned opt-in FP8 cache/indexer precision lane：窄 mini-owned FP8 indexer cache/logits slice 质量通过但性能失败，4096/128 从 exact control `37.9237` 降到 `29.6691 output tok/s`，停止该 slice。 |
| TARGET 07.51 | `prompts/TARGET_07.51_dsv4_sm80_vllm_fp8_backend_parity.md` | completed | 已隔离 vLLM 原生 FP8 indexer 与 `fp8_ds_mla` gather/dequant backend。vLLM FP8 paged decode logits 在 batch16/history4096 为 `0.1529 ms`，明显快于 mini bf16 logits `0.3076 ms` 和 mini FP8 logits `1.3072 ms`；结论是下一步 port/adapt vLLM FP8 indexer backend。完整 `fp8_ds_mla` KV cache 暂缓，且不要照搬 SM80 会卡住的 standalone `quantize_and_insert_k_cache`。 |
| TARGET 07.52 | `prompts/TARGET_07.52_dsv4_sm80_vllm_fp8_indexer_backend_port.md` | completed | 已将 vLLM-aligned FP8 paged indexer backend 作为 opt-in 路径 port/adapt 到 mini。large-shape paged logits `0.1845 ms`，接近 vLLM `0.1529 ms`；text smoke pass；4096/1024/batch4 达到 `73.67 output tok/s`，比历史 exact `68.81` 约 `+7%`，但仍远低于 `114.07` 和 vLLM `~202`。 |
| TARGET 07.53 | `prompts/TARGET_07.53_dsv4_sm80_post_fp8_indexer_reprofile.md` | completed | 在 FP8 indexer 成功后重新抓 mini profile，并和 vLLM 对比剩余 top buckets。结论：decode envelope 中 projection/GEMM `1.7973 s`，graph/runtime/copy/cat/index `1.6170 s`，elementwise graph nodes `1.3583 s`；graph/layout cluster 合计 `2.9752 s` / `45.50%`，下一步先做 graph/layout replay deforestation，projection/GEMM 作为 pivot。 |
| TARGET 07.54 | `prompts/TARGET_07.54_dsv4_sm80_graph_layout_replay_deforestation.md` | completed | fused FP8 activation fake-quant PoC 命中 profile gate：graph/layout cluster 从 `2.9752 s` 降到 `1.8271 s`（`-38.59%`），4096/128 从 `41.66` 到 `43.07 output tok/s`，4096/1024 从 `73.67` 到 `87.08 output tok/s`；剩余 graph/layout 与 projection/GEMM 并列。 |
| TARGET 07.55 | `prompts/TARGET_07.55_dsv4_sm80_remaining_graph_layout_or_projection_pivot.md` | completed | 对 07.54 后剩余 graph/layout 做重新归因；direct-copy 太分散，BF16 copy 太小，CatArray/index/gather 只有堆叠后才刚过 gate，pow/mean/mul 缺少单一边界；结论是不要继续泛 graph/layout，pivot 到 projection/GEMM backend parity。 |
| TARGET 07.56 | `prompts/TARGET_07.56_dsv4_sm80_low_cost_graph_layout_compile_preflight.md` | completed | projection/GEMM parity 前的短 preflight：实现 opt-in static scale cache，microbench 消除 scale cast/copy 但 4096/128 只从 `43.0685` 到 `43.2194 output tok/s`（`+0.35%`）；结论是不继续低成本 graph/layout，小修上下文交给 projection/GEMM parity。 |
| TARGET 07.57 | `prompts/TARGET_07.57_dsv4_sm80_projection_gemm_backend_parity.md` | next todo | 对 `1.7968 s` projection/GEMM bucket 做 owner-level attribution，并和 vLLM `QuantFP8`、scaled FP8 quant、linear/custom-op、`deepseek_v4_fp8_einsum`、SM80 `wo_a` BMM/reference 路径对标；之后只选择一个证据支持的 PoC，如 small-M kernel retune、vLLM 边界 adaptation 或 quant+GEMM fusion。 |
| TARGET 08 | `prompts/TARGET_08_radix_prefix_dsv4.md` | planned | DSV4 radix prefix cache v2，等 TARGET 07 的非前缀高性能路径稳定后再推进。目标是 shared-prefix 请求跳过 cached-prefix prefill，且 DSV4 多组件 KV cache/SWA/压缩状态引用和 eviction 正确。 |

Target 6 baseline命令

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback v0_bf16 \
  --page-size 256 \
  --output-dir /tmp/dsv4_sm80_target06_tp8 \
  --keep-going
  ```

Target 6 text correctness smoke命令

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback v0_bf16 \
  --output /tmp/dsv4_text_smoke.json
  ```

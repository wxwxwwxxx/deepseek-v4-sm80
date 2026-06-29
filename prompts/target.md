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

# New Thread Prompt for TARGET 07.2

Archived note: TARGET 07.2 now has recorded artifacts in
`performance_milestones/target07_comm_graph/` and should not be used as the
default next thread. Use `prompts/THREAD_PROMPT_TARGET_07.md` for the current
TARGET 07 cycle unless you specifically need to replay or audit 07.2.

你好，请继续推进 `/workspace/mini-sglang` 中 DeepSeek V4 Flash 在
A100/sm80 上的性能追赶工作。总目标见 `prompts/target.md`，当前大目标是
`prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`：在 TP8、page/block size
256、4096 input / 1024 output / batch4 下超过旧 vLLM-based serving baseline
`114.07 output tok/s`，同时默认路径保持 exact。

现在请进入 `prompts/TARGET_07.2_dsv4_sm80_comm_cuda_graph.md`。不要回到
TARGET 07.1 继续复测，也不要一上来做 MoE V2 或 activation quantization。

## 必读背景

请先阅读：

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.2_dsv4_sm80_comm_cuda_graph.md`
- `performance_milestones/target07_vllm_gap/RESULTS.md`
- `performance_milestones/target07_vllm_gap/EXECUTION_DIFF.md`
- `performance_milestones/target07_vllm_gap/README.md`

重要结果：

- fair mini 4096/1024/bs4：`10.5768 output tok/s`
- fair vLLM 4096/1024/bs4：`201.874 output tok/s`
- 旧 vLLM-based serving baseline：`114.07 output tok/s`
- fair vLLM / mini：`19.09x`
- 旧 baseline / mini：`10.78x`
- fair vLLM 4096/128/bs4：`80.9050 output tok/s`
- fair mini 4096/128/bs4 default prefill：`5.5071 output tok/s`
- mini `MAX_EXTEND_TOKENS=4096` 没有缩小 gap：`5.3471 output tok/s`

注意：当前 fair vLLM 不是未经优化的 stock vLLM，它已经包含用户优化过的
fp8 indexer kernel，因此 `201.874 output tok/s` 应视为强 stretch reference。
TARGET 07 的硬性第一胜利线仍是旧 baseline `114.07 output tok/s`；超过 fair
vLLM 是后续 stretch goal。

当前 Nsight 结论：

- mini rank0 4096/128 default prefill：`6,663,421` CUDA kernels，
  `22,528` NCCL kernels，CUDA graph events 为 `0`
- vLLM total 4096/128：`124,480` CUDA kernels，`16` NCCL kernels，
  CUDA graph events 为 `7,200`
- mini 的主要问题是通信碎片化、DSV4 decode 没有 CUDA graph replay、
  PyTorch 小 kernel 爆炸，以及 grouped FP4 MoE kernel 本身仍重。

## 当前任务

第一阶段先做通信与 CUDA graph，不做 MoE V2。

1. 检查当前代码和 artifact。
   - 模型路径：`/models/DeepSeek-V4-Flash`
   - mini benchmark：`benchmark/offline/deepseek_v4_perf_matrix.py`
   - TARGET 07.1 artifact：`performance_milestones/target07_vllm_gap/`
   - vLLM 源码：`/workspace/vllm-dsv4-docker`
   - vLLM venv：`/workspace/venvs/vllm-dsv4`，不要用于 mini-sglang

2. 添加通信 observability，不改变默认语义。
   - 给 `DistributedCommunicator` 或调用边界增加可选 semantic label。
   - 至少标注：
     - embedding all-reduce
     - attention/row-parallel projection all-reduce
     - routed expert all-reduce
     - shared expert all-reduce
     - V1 reduce-once MoE all-reduce
     - lm_head all-gather
   - 在 benchmark report 中记录 label、op、dtype、shape、bytes、count。
   - 保持 V1 MoE late reduce invariant：routed + shared 本地相加后只做一次
     MoE all-reduce，不要退回两个 reduce。

3. 修复并评估 PyNCCL exact path。
   - 检查 `python/minisgl/kernel/pynccl.py` 和
     `python/minisgl/kernel/csrc/src/pynccl.cu`。
   - 已知旧 blocker：PyNCCL 到达 DSV4 forward 后在
     `lm_head.linear()` all-gather 附近失败；本地 dtype map 缺 fp32 支持。
   - 增加或修复 bf16/fp16/fp32 all-reduce 和 all-gather 的 TP 测试。
   - 通过 TP8 DSV4 text smoke 后，再加入 benchmark variant，例如
     `v1_moe_pynccl`。
   - 只有 correctness 和性能都成立，才考虑让 PyNCCL 成为 DSV4 TP8 guarded
     default 或 benchmark path。

4. 安全启用 DSV4 decode CUDA graph。
   - 当前 `python/minisgl/engine/engine.py` 会强制 DSV4
     `cuda_graph_bs=[]`、`cuda_graph_max_bs=0`。
   - 不要直接粗暴打开默认路径。先加显式 flag 或 variant-controlled allowlist。
   - 只捕获 decode，prefill 保持 eager。
   - 从 capture sizes `[1,2,4]` 开始。
   - 确保 DSV4 attention metadata、page table、positions、input ids、
     output locations 等要么是稳定 graph input，要么 replay 前 copy 到固定
     capture buffer。
   - 如果 capture 失败，记录精确 blocker：哪个 op、哪个 tensor shape、
     哪个 mutable metadata 或哪个 allocator/stream 问题。

5. 对照 vLLM，但不要依赖 vLLM runtime。
   - 可参考 vLLM 的 CUDA graph dispatcher、communication custom ops、
     custom all-reduce、DeepSeek V4 attention 边界。
   - 不要把 vLLM 作为 mini runtime dependency。
   - 不要默认移植 vLLM sm80 reference sparse prefill；这条路径之前有 OOM 风险。
   - 不做 activation quantization；bf16-direct exact 仍是第一阶段默认路线。
   - 模型中原有 fp32 计算不要静默降精度；TF32 只能作为显式实验。

6. 每个阶段都要复测。
   - 先跑 correctness/text smoke。
   - 再跑 4096/128/bs4 nsys，对比 kernel count、runtime count、NCCL count、
     CUDA graph events。
   - 最后跑 4096/1024/bs4 macro benchmark。
   - 新 artifact 放到 `performance_milestones/target07_comm_graph/` 或
     `performance_milestones/target07_vllm_gap/` 下一个清晰子目录中。

## Done Criteria

TARGET 07.2 完成时，应能回答：

- 哪些 semantic communication site 贡献了最多 count/bytes？
- PyNCCL exact path 是否正确？如果不正确，精确 blocker 是什么？
- DSV4 decode CUDA graph 是否能 replay `[1,2,4]`？如果不能，精确 blocker 是什么？
- best exact variant 在 4096/128/bs4 nsys 和 4096/1024/bs4 macro 下分别是多少？
- 下一步应该进入 TARGET 07.3 MoE V2，还是继续补 communication/CUDA graph？

请保持实现范围小而可测。每次优化都要保留 exact 默认语义，且不要回滚用户已有改动。

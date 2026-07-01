# New Thread Prompt for TARGET 07

你好，请继续推进 `/workspace/mini-sglang` 中 DeepSeek V4 Flash 在 A100/sm80
上的性能追赶工作。我们的总目标是 `prompts/target.md`，当前大目标是
`prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`：在 TP8、page/block size
256、4096 input / 1024 output / batch4 下超过旧 vLLM-based 框架的
`114.07 output tok/s` 基线，同时默认路径保持 exact。

请先从 `prompts/TARGET_07.1_dsv4_sm80_fair_rebench_vllm_diff.md` 开始，不要一上来直接实现 MoE V2 或 CUDA graph。第一步目标是做公平复测和 vLLM 执行路径对标：

- mini 当前主要证据在 `performance_milestones/v1_moe/README.md`；
- vLLM 环境和脚本在 `performance_milestones/vllm/README.md`；
- vLLM 源码在 `/workspace/vllm-dsv4-docker`；
- vLLM 虚拟环境在 `/workspace/venvs/vllm-dsv4`，不要用于 mini-sglang；
- 模型路径是 `/models/DeepSeek-V4-Flash`；
- mini benchmark 是 `benchmark/offline/deepseek_v4_perf_matrix.py`；
- vLLM benchmark shim 是 `performance_milestones/vllm/scripts/run_vllm_deepseek_v4_matrix.py`。

重要背景：

- mini V1 MoE 已经显著改善 V0，但 4096/1024/batch4 仍只有约 `10.51 output tok/s`；
- 用户提供的旧 vLLM serving 基线是 `114.07 output tok/s`、TTFT `123.21ms`、TPOT `15.68ms`；
- 现有 4096/128/batch4 nsys 不是完全公平：vLLM 用了 warmup=1、chunked prefill=4096、CUDA graph sizes 1/2/4，而 mini 用了 warmup=0，且 DSV4 CUDA graph 被禁用；
- 已知 mini profile 中最大问题是 NCCL all-reduce、grouped FP4 MoE、PyTorch 小 kernel 爆炸，以及没有 DSV4 CUDA graph replay。
- 精度路线图是：第一阶段优先 bf16 Tensor Core，不做 activation quantization；模型中原有 fp32 计算不要静默降精度，TF32 只能作为显式实验；第二阶段才评估 fp8/fp4 activation quantization，并可优先参考 vLLM；第三阶段再做 INT8 Tensor Core opt-in。

请在新 thread 中：

1. 先阅读 `TARGET_07` 和 `TARGET_07.1`，再检查现有脚本和 artifact。
2. 生成或修正公平复测脚本/命令，输出到 `performance_milestones/target07_vllm_gap/`。
3. 对照 vLLM 的 `FusedMoE`、CUDA graph dispatcher、custom all-reduce、communication custom ops、DeepSeek V4 attention 路径，写出 mini/vLLM execution diff。
4. 对每个 vLLM 设计点明确结论：`port`、`adapt`、`reject` 或 `defer`。
5. 完成后给出下一步应该进入 `TARGET_07.2` 还是先补测的判断。

请注意：我们希望尽量学习和复用 vLLM 中已经证明有效的设计，避免重复造一个更慢的轮子；但不要把 vLLM 作为 mini 的 runtime dependency，也不要默认移植 vLLM sm80 reference sparse prefill 那条已经触发过 OOM 的路径。
同时，第一阶段不要求和 vLLM 的精度实现完全对齐；优先把 mini 的 bf16-direct exact 路径做好。

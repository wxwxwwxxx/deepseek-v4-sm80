# New Thread Prompt for TARGET 07 Current Cycle

你好，请继续推进 `/workspace/mini-sglang` 中 DeepSeek V4 Flash 在 A100/sm80
上的性能追赶工作。总目标见 `prompts/target.md`，当前大目标是
`prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`：在 TP8、page/block size
256、4096 input / 1024 output / batch4 下超过旧 vLLM-based serving baseline
`114.07 output tok/s`，默认路径保持 exact。

当前状态：

- TARGET 07.1 已完成公平 mini/vLLM 复测，artifact 在
  `performance_milestones/target07_vllm_gap/`。
- TARGET 07.2 已完成通信/graph 轨迹记录，artifact 在
  `performance_milestones/target07_comm_graph/`。best exact 4096/1024/bs4
  约为 `25.3 output tok/s`。
- TARGET 07.25 已完成 mini/vLLM 子图 parity，artifact 在
  `performance_milestones/target07_subgraph_parity/`。瓶颈排序是：
  MoE routed/execution boundary、attention/indexer/cache、
  scheduling/stream overlap、communication、precision、HC/final。

现在请进入：

- `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`

请先阅读：

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.3_dsv4_sm80_moe_v2_exact.md`
- `performance_milestones/target07_subgraph_parity/README.md`
- `performance_milestones/target07_comm_graph/README.md`
- `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md`

重要约束：

- 本阶段只做 exact MoE V2，不做 activation quantization、不做 INT8、不把
  vLLM 作为 runtime dependency。
- 默认精度路线是 bf16-direct。模型中原有 fp32 计算不要静默降精度；TF32
  只能作为单独显式实验。
- vLLM 的 FusedMoE runner、route metadata、workspace/finalize、shared expert
  scheduling 可以作为优先参考；vLLM MXFP4/FP8 精度语义先 defer 到
  `prompts/TARGET_07.4_dsv4_sm80_precision_lanes.md`。
- vLLM `DeepseekV4MegaMoEExperts` 在 sm80 上不是路线，不要移植。

工作重点：

1. 建立 mini-side MoE execution plan：route metadata、token/expert layout、
   workspace ownership、finalize/reduce policy。
2. 对照 vLLM `FusedMoE` 的 prepare/fused-experts/finalize 边界，选择可以
   adapt 到 mini exact lane 的部分。
3. 优先减少 MoE routed experts 的 W13/W2 时间、workspace/materialization、
   route/finalize 小 kernel，以及 reduce boundary 的不确定性。
4. shared expert overlap 只有在 profile 证明它仍然重要时再做。
5. 每个 serious cut 后跑 correctness/text smoke、4096/1024 macro，以及
   4096/128 short profile 或等价 profile。

Stop rules：

- 如果 4096/1024/batch4 exact output throughput 超过 `114.07 tok/s` 且 TP8
  page-size-256 text smoke 通过，停止性能扩张并记录胜利线。
- 如果 MoE routed W13/W2 不再是 top-two bottleneck，停止 07.3，进入
  `prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md`。
- 如果一次 serious MoE cut 让 macro 提升至少 `1.3x`，或让 MoE W13/W2
  summed kernel time 降低至少 `2x`，停止继续钻 MoE，先做 07.35 re-parity。
- 如果连续两次 MoE cut 都低于 `5%` macro gain 且低于 `10%` routed-MoE
  subgraph gain，停止本 thread 并记录原因。
- 如果下一步明显属于 attention/cache、communication、precision 或普通 graph
  cleanup，不要在 07.3 里继续做，转 07.35 重新排序。

最终请更新：

- `performance_milestones/target07_moe_v2/` 或本 target 选择的 milestone
  目录；
- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`；
- 必要时更新 `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md`。

完成 07.3 后，默认下一步是
`prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md`，除非已经超过
`114.07 tok/s` 胜利线。

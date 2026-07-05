# TARGET 08.34 MoE Marlin WNA16 Cache Lifecycle

结论：TARGET 08.33 看到的 warmup `model.forward()` 17-18 GiB/rank 大额显存增长，可以由 A100 victory 路径的 MoE `marlin_wna16` routed expert lazy repack 完整解释。不是实际 `torch.cuda.graph` capture 块内的私有池增长。

## Direct Answers

1. **`marlin_wna16` lazy repack 是否解释了 17-18 GiB/rank？是。** 当前 `dsv4_sm80_a100_victory_prefix_routeb_lifetime` preset 在未显式设置 expert backend 时选择 `marlin_wna16`。rank0 current run 的 warmup forward allocated delta 是 `19,126,168,064` bytes = `17.8126` GiB；prebuild 后 residual 是 `729,950,720` bytes = `0.6798` GiB；两者差值正好是 `18,396,217,344` bytes = `17.1328` GiB，即 43 层 Marlin WNA16 routed expert cache 理论大小。

2. **prebuild 后 warmup forward 跳变是否消失？是。** `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1` 后，warmup forward delta 从 `17.8126` GiB 降到 `0.6798` GiB，与 forced `grouped_fp4` 的 `0.7225` GiB 同量级；实际 CUDA graph capture 阶段仍只有 `1,024` bytes allocated delta。

3. **prebuild 对 KV capacity planning 的影响：修正了 KV 预算口径。** Engine 已在 `_determine_num_pages()` 前调用 `prepare_for_cuda_graph_capture()`。prebuild 不释放原始 FP4 时，KV planning 会提前看到 Marlin cache 的 `17.1328` GiB/rank 持久占用，理论上少给约 `400.10` 个 page = `102,426` tokens，避免把随后 lazy cache 会吃掉的显存误分配给 KV。本实验固定 `--num-pages 128`，所以实际 KV pages 不随 run 改变。

4. **原始 FP4 expert weights 是否可以安全释放、能省多少？运行期可行，但不应默认启用。** `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1` + `MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1` smoke 通过，`decode_len=2` 下 CUDA graph replay_count=`1`、eager_decode_count=`0`。它能释放 `18,396,217,344` bytes = `17.1328` GiB/rank 原始 FP4 expert weights/scales，抵消 prebuild cache 的额外常驻显存。不过删除后同一 Engine 不能无缝切回 `grouped_fp4`/fallback，也影响 reload、state_dict 和 rollback 语义；当前只建议 opt-in。

5. **建议：promote prebuild lifecycle，release 保持 opt-in；无需继续把 17-18 GiB 主因归咎其他 owner。** Marlin WNA16 cache lifecycle 应对齐 SGLang/vLLM 的 post-load/pre-forward packing 设计。建议先把 prebuild 作为 `marlin_wna16` backend 的 promotion candidate，在 A100 victory 路径跑完更宽 bucket/text regression 后默认启用；原始 FP4 release 保留 opt-in，并继续补 reload/fallback/rollback 语义测试。剩余 `0.68-0.72` GiB warmup residual 可另开 owner attribution，但它不是 17-18 GiB spike 主因。

## Backend Check

命令见 `COMMANDS.md`。观测：

```text
backend marlin_wna16
explicit_env None
```

所以当前 A100 victory preset 的 MoE expert backend 确实是 `marlin_wna16`。

## Theoretical Ledger

DeepSeek-V4-Flash config：`hidden_size=4096`，`moe_intermediate_size=2048`，TP=8，所以 `local_intermediate=256`；`n_routed_experts=256`，`num_hidden_layers=43`。

| Item | Bytes/rank | GiB/rank |
| --- | ---: | ---: |
| per-layer raw/repacked `w13_weight` | 268,435,456 | 0.2500 |
| per-layer raw/repacked `w2_weight` | 134,217,728 | 0.1250 |
| per-layer raw/repacked `w13_scale` | 16,777,216 | 0.0156 |
| per-layer raw/repacked `w2_scale` | 8,388,608 | 0.0078 |
| per-layer total | 427,819,008 | 0.3984 |
| 43-layer raw total | 18,396,217,344 | 17.1328 |
| 43-layer Marlin repacked total | 18,396,217,344 | 17.1328 |
| raw + repacked during prebuild-without-release | 36,792,434,688 | 34.2656 |

`gptq_marlin_repack` output layout has the same byte size as mini 的 packed int8 source tensors for these WNA16 FP4 shapes；scales are also byte-identical in total. Full ledger is in `summaries/moe_marlin_lifecycle_summary.md`.

With page size 256 and indexer fp8 cache enabled, `17.1328` GiB/rank is about `400.10` KV pages or `102,426` KV tokens.

## A/B Results

All runs used one bucket (`--cuda-graph-bs 16`, smoke, TP=8). Raw logs live under `raw/`; parsed summary lives under `summaries/`.

| Run | Backend / opt-in | Status | Marlin rank0 cache rows | Marlin rank0 repacked | Warmup forward rank0 allocated delta | Actual graph capture rank0 allocated delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `current_marlin_bs16` | current victory `marlin_wna16`, lazy | pass | 43 | 17.1328 GiB | 17.8126 GiB | 1,024 bytes |
| `forced_grouped_fp4_bs16` | forced `grouped_fp4` | pass | 0 | 0 | 0.7225 GiB | 1,024 bytes |
| `prebuild_marlin_bs16` | `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1` | pass | 43 | 17.1328 GiB | 0.6798 GiB | 1,024 bytes |
| `prebuild_release_marlin_bs16_decode2` | prebuild + release original FP4 | pass | 43 | 17.1328 GiB | 0.6798 GiB | 1,024 bytes |

Key equality:

```text
current warmup allocated delta - prebuild warmup allocated delta
= 19,126,168,064 - 729,950,720
= 18,396,217,344 bytes
= 17.1328 GiB/rank
= theoretical 43-layer Marlin WNA16 cache
```

Owner-level warmup instrumentation confirms the same shape: in current lazy Marlin, rank0 top warmup deltas are `moe.routed_experts after` for layers 0..42, each about `427,950,080` bytes = `0.3986` GiB. In prebuild, grouped, and release runs, MoE no longer dominates the top warmup deltas.

## Model Prepare Report

`model_prepare_report_rank0["moe_marlin_wna16_cache"]`:

| Run | enabled | backend | prebuild | release_original | layers_cached | persistent bytes | source bytes | released bytes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| current | false | `marlin_wna16` | false | false | 0 | 0 | 0 | 0 |
| forced grouped | false | `grouped_fp4` | false | false | 0 | 0 | 0 | 0 |
| prebuild | true | `marlin_wna16` | true | false | 43 | 18,396,217,344 | 18,396,217,344 | 0 |
| prebuild + release | true | `marlin_wna16` | true | true | 43 | 18,396,217,344 | 18,396,217,344 | 18,396,217,344 |

The release run also captured bs16 and replayed decode once:

```text
captured_bs=[16]
replay_count=1
eager_decode_count=0
```

## Implementation

Added opt-in instrumentation and lifecycle controls:

- `MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1`: JSONL around `prepare_moe_mxfp4_weights()` with before/after allocated/reserved/free memory, source tensor bytes, repacked tensor bytes, owner, rank.
- `MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1`: JSONL around warmup `model.forward()` owner/layer boundaries.
- `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1`: builds all routed expert Marlin WNA16 caches during `prepare_for_cuda_graph_capture()`, before KV capacity planning.
- `MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1`: after prebuild, deletes original routed FP4 expert tensors. This requires prebuild and `marlin_wna16`; other backends raise explicitly if raw tensors are missing.

Main code paths:

- `python/minisgl/kernel/marlin_wna16.py`: opt-in cache prepare memory JSONL.
- `python/minisgl/models/deepseek_v4.py`: per-layer owner memory records, MoE cache prebuild/report/release, prepacked-only dispatch after release.
- `python/minisgl/kernel/deepseek_v4.py`: lazy wrapper now reports owner/cache state and shares a prepacked dispatch helper.
- `python/minisgl/engine/graph.py`: wraps CUDA graph warmup `model.forward()` with warmup memory context.
- `benchmark/offline/deepseek_v4_perf_matrix.py`: preserves lifecycle debug envs and adds a single forced `grouped_fp4` diagnostic variant.
- `python/minisgl/utils/dsv4_memory_debug.py`: common JSONL/memory/tensor summary helpers.

## Upstream Lifecycle Comparison

SGLang and vLLM both put heavyweight quantized-weight processing on the load/post-load path, not first routed expert forward:

- SGLang `DefaultModelLoader.load_weights_and_postprocess()` calls each module's `quant_method.process_weights_after_loading(module)` after model weights are loaded. Main branch reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py
- SGLang `Mxfp4MarlinMoEMethod.process_weights_after_loading()` prepares MXFP4 experts for Marlin and calls `prepare_moe_mxfp4_layer_for_marlin(layer)` before serving forward uses `MarlinMoeQuantInfo`. Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
- vLLM `process_weights_after_loading()` has the same loader hook, and its Marlin FP4 helper performs `ops.gptq_marlin_repack(...)` in the prepare helper. References: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/utils.py and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py

Recommendation: mini-sglang should align lifecycle with those designs: heavyweight MoE Marlin packing should be explicit in model prepare/post-load and visible to KV capacity planning. Direct code reuse is not obviously drop-in because mini's DSV4 raw packed tensor shapes and owner/reporting contracts differ, but the lifecycle should match.

## Validation

```text
python -m py_compile python/minisgl/utils/dsv4_memory_debug.py \
  python/minisgl/kernel/marlin_wna16.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/graph.py

python -m py_compile benchmark/offline/deepseek_v4_perf_matrix.py
python -m py_compile performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/scripts/summarize_moe_marlin_lifecycle.py
```

Runtime validation:

- current Marlin single-bucket smoke: pass
- forced grouped_fp4 single-bucket smoke: pass
- prebuild Marlin single-bucket smoke: pass
- prebuild + release-original FP4 smoke with one graph decode replay: pass

## Artifacts

- Commands: `COMMANDS.md`
- Raw JSONL: `raw/`
- Parsed theoretical/runtime summary: `summaries/moe_marlin_lifecycle_summary.md`
- Parsed machine-readable summary: `summaries/moe_marlin_lifecycle_summary.json`

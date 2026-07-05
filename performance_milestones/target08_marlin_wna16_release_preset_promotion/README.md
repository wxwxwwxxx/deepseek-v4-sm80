# TARGET 08.35 Marlin WNA16 Release Preset Promotion

结论：release preset 的命名、env 展开、两阶段 prebuild/release、reporting 和 fail-closed 语义已经落地，但 **不建议 promote**。当前 TP8 text smoke 显示 `dsv4_sm80_a100_victory_marlin_release` 在释放 raw routed FP4 expert weights/scales 后会稳定产生异常文本；这命中本 target 的硬阻断条件。

## Direct Answers

1. **release preset 的准确 env 展开是什么？**
   `dsv4_sm80_a100_victory_marlin_release` 展开为：

   ```text
   MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
   MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
   MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
   MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
   ```

   prefix 版本 `dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release` 在上述基础上还展开：

   ```text
   MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
   MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
   MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
   ```

   并要求 benchmark/runtime CLI 使用：

   ```text
   --enable-dsv4-radix-prefix-cache
   --enable-dsv4-component-loc-ownership
   ```

2. **是否恢复约 17.13 GiB/rank？**
   model prepare ledger 显示 release preset 释放 `18,396,217,344` bytes = `17.1328` GiB/rank，43 层全部 release。固定 `--num-pages 128` smoke 中，rank0 free memory after initialization 从 prebuild-only 的 `38.36` GiB 增至 release 的 `54.01` GiB，实际观测增加约 `15.65` GiB；差值包含 allocator/runtime/capture 口径差异。按 TARGET 08.34 的 page-size 256 ledger，这相当于约 `400.10` KV pages 或 `102,426` tokens/rank 的理论 headroom。

3. **graph replay、text smoke、prefix smoke 是否通过？**
   graph capture/replay 机械 gate 在 release smoke 中通过：requested/captured bs 都是 `[1,2,4,8,16]`，`replay_count=63`，`eager_decode_count=0`。但 text smoke 不通过，status=`warn`，三条输出均 `looks_sane=false`。prefix smoke 未继续执行，因为 non-prefix release text correctness 已经失败，按 target 阻断 promotion。

4. **fallback/backend switch 是否 fail closed 且错误信息清晰？**
   是。单元测试覆盖 grouped/fallback backend switch after release，并验证明确报错：

   ```text
   Marlin WNA16 release preset has released raw routed expert weights; fallback/grouped_fp4 backend is unavailable in this Engine. Use the non-release preset or recreate the Engine with release disabled.
   ```

5. **macro 性能是否中性或更好？**
   未执行 macro。原因是 release text smoke 已经产生文本异常，继续跑 4096x128 或 4096x1024 只会测量一个不正确 runtime 的吞吐，不应作为 promotion 证据。

6. **是否建议将 release preset 作为新的高显存效率 milestone？**
   不建议。建议保留命名 preset 和 fail-closed 安全语义作为后续修复基础，但当前 release preset 不应成为新的高显存效率 milestone。可 promotion 的方向仍是 prebuild lifecycle；release 需要先定位文本异常。

## Implementation

已完成的代码侧改动：

- `benchmark/offline/deepseek_v4_perf_matrix.py` 和 `benchmark/offline/deepseek_v4_text_smoke.py` 新增：
  - `dsv4_sm80_a100_victory_marlin_prebuild`
  - `dsv4_sm80_a100_victory_marlin_release`
  - `dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release`
- release preset 固定 `marlin_wna16 + prebuild + release`，并且不再把 loose release env 作为跨 variant preserved env 泄漏到其它 preset。
- `python/minisgl/models/deepseek_v4.py` 将 release 改成全模型两阶段流程：先所有 layer prebuild 成功，再统一释放 raw weights；prebuild 失败时不会释放任何 raw tensor。
- release 后同一 Engine 标记为 `marlin_wna16_prepacked_only`，`forward()` 在 raw tensor 缺失且 backend 不是 `marlin_wna16` 时 fail closed。
- `state_dict()` 路径不会因已释放 raw expert tensors 崩溃。

## Runtime Results

All runtime runs used TP=8, `/models/DeepSeek-V4-Flash`, page size `256`, fixed `--num-pages 128`, `--cuda-graph-bs 1 2 4 8 16`.

| Run | Variant | Status | Text sanity | Marlin cache | Released raw | Init free memory rank0 | Graph captured bs | Replay/eager |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `text_smoke_nonrelease_baseline` | `dsv4_sm80_a100_victory` | pass | pass | 0 | 0 | 55.49 GiB | `[16,8,4,2,1]` | 9 / 0 |
| `text_smoke_prebuild_only` | `dsv4_sm80_a100_victory_marlin_prebuild` | pass | pass | 17.1328 GiB | 0 | 38.36 GiB | `[16,8,4,2,1]` | 9 / 0 |
| `text_smoke_release_sync` | `dsv4_sm80_a100_victory_marlin_release` | warn | fail | 17.1328 GiB | 17.1328 GiB | 54.01 GiB | `[16,8,4,2,1]` | 63 / 0 |

Release smoke model report:

```text
layers_cached=43
total_persistent_bytes=18,396,217,344
total_source_bytes=18,396,217,344
total_released_original_bytes=18,396,217,344
release_runtime_policy=marlin_wna16_prepacked_only
```

Release output sanity failure sample:

```text
2 + #####  #  #  #  #  ___________________________________________________________________________
# |
```

Baseline and prebuild-only generated sane text for the same prompts:

```text
2 + 2 等于 4。
The sky is blue on a clear day.
杭州是风景如画的历史文化名城。
```

## Promotion Decision

Promotion is rejected for this target. The release preset meets the memory ledger and fail-closed requirements, and graph replay does not fall back to eager, but correctness fails. The target explicitly says release must not cause text corruption; therefore:

- do not promote `dsv4_sm80_a100_victory_marlin_release`;
- do not promote the prefix release preset;
- do not run or publish macro throughput as release evidence until correctness is fixed;
- keep the non-release/prebuild path as the viable lifecycle improvement.

## Validation

Commands are listed in `COMMANDS.md`.

Passed local checks:

```text
python -m py_compile python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

pytest -q tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

Result: 76 tests passed.

## Artifacts

- Raw smoke JSON/logs: `raw/`
- Main release evidence: `raw/text_smoke_release_sync.json`
- Baseline evidence: `raw/text_smoke_nonrelease_baseline.json`
- Prebuild-only evidence: `raw/text_smoke_prebuild_only.json`
- Commands: `COMMANDS.md`

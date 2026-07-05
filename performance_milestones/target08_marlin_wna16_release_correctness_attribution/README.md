# TARGET 08.36 Marlin WNA16 Release Correctness Attribution

结论：**NO-GO for `dsv4_sm80_a100_victory_marlin_release` promotion**。

TARGET 08.35 的三路 smoke 已复用，并补跑了 graph/eager、logit parity、MoE packed-cache micro parity、release lifetime A/B、cache integrity 和 activation divergence 诊断。结果显示：release 失败不是 Marlin WNA16 prebuild 本身，也不是 prepacked MoE runtime path 本身；当前最可能的 blocker 是 **在 KV/graph/warmup 等后续分配前物理释放 routed FP4 expert weight storage 后，某个全模型 decode 路径仍间接依赖这批 raw weight storage 的生命周期或地址稳定性**。

因此 release preset 继续 blocked。安全路线仍是 `dsv4_sm80_a100_victory_marlin_prebuild`，不要在 text sanity 已知失败时继续跑 4096x128 / 4096x1024 macro。

## Direct Answers

1. **三路 08.35 text smoke 结论是否复用/确认？**

   是。复用 `../target08_marlin_wna16_release_preset_promotion/raw`：

   | Variant | Text smoke | Generated token prefix |
   | --- | --- | --- |
   | `dsv4_sm80_a100_victory` | pass/pass | sane baseline |
   | `dsv4_sm80_a100_victory_marlin_prebuild` | pass/pass | same sane tokens as baseline |
   | `dsv4_sm80_a100_victory_marlin_release` | warn/warn | first few tokens plausible, then token `0` flood |

   The release run still reports `17.1328 GiB/rank` released, so the memory recovery is real but unsafe.

2. **graph replay 是不是 primary owner？**

   不是。release with graph fails, but release eager/no-graph also fails. The graph run with greedy sampling capture disabled still corrupts logits; this rules out greedy-sample graph replay as the primary owner.

3. **first logit divergence 在哪里？**

   `logit_parity_ladder.md` shows prebuild-only and release are bit-identical through:

   - prefill last logits;
   - decode step 1 logits;
   - decode step 2 logits.

   First token divergence is `uid0:decode_step_3`: prebuild token `20`, release token `0`. At that step, `max_abs_diff=46.1453`, `mean_abs_diff=3.2487`, and release top-k probabilities collapse to a near-uniform tiny distribution around tokens `0..4`. This is model/logit corruption, not a text parser artifact.

4. **MoE Marlin WNA16 packed-cache 本身是否不一致？**

   The micro parity probe did not reproduce an MoE output mismatch. On rank0 layers `0/21/42`, raw-present vs force-prepacked, raw-present vs released-same-cache, and force-prepacked vs released-same-cache are all `0/0` max/mean abs diff, even after `768 MiB` allocator pressure. Grouped oracle vs Marlin differs only by expected small quant/path noise.

5. **release lifetime A/B 排除了什么？**

   | A/B case | Result | Meaning |
   | --- | --- | --- |
   | force-prepacked raw-present | pass/pass | prepacked runtime path is safe while raw attrs/storage remain present |
   | keep-hidden-ref | pass/pass | deleting normal attrs is safe if original tensor storage remains alive |
   | release-after-capture | pass/pass | release can be safe after KV allocation and graph capture |
   | weights-only | warn/warn | freeing large expert weight storage is enough to trigger corruption |
   | scales-only | pass/pass | scale storage alone is not the trigger |
   | layer0, layers0-7 | pass/pass | small releases are tolerated |
   | layers0-15, layers0-20, layers21-42 | warn/warn | failure follows large physical release, not one specific layer |

   This strongly separates branch-change/delattr from physical storage lifetime. The threshold is between about `3.1875` and `6.3750 GiB/rank` released in this setup.

6. **packed cache integrity 是否稳定？**

   Yes for the sampled layers/rank. `cache_integrity_summary.md` records stable `data_ptr`, shape/dtype/stride, finite/checksum summaries, and zero cache tensor changes after release through `empty_cache`, graph capture, and first graph replay for layers `0/21/42` on rank0.

7. **first divergent owner 是谁？**

   Two levels were observed:

   - Logit-level first visible divergence: `uid0:decode_step_3`.
   - Activation-level eager diagnostic first recorded mismatch: `layer2.indexer_select.logits`, then `layer2.attention_backend.merged_attention_output_before_wo`, then downstream MoE inputs/outputs.

   This points the first full-model symptom toward attention/indexer state after release, but the A/B evidence says the root owner is still raw expert **weight storage lifetime after early physical release**, not a proven MoE Marlin packed-cache math bug.

## Evidence Tables

Main summaries:

- `smoke_graph_split_matrix.md`
- `logit_parity_ladder.md`
- `moe_micro_parity.md`
- `release_lifetime_ab.md`
- `cache_integrity_summary.md`
- `activation_divergence.md`

Raw artifacts:

- `raw_08_35_smoke/` symlink to TARGET 08.35 smoke artifacts.
- `raw/text_smoke_release_graph_mt{1,2,4,16}.json`
- `raw/text_smoke_release_eager_mt16_rerun.json`
- `raw/text_smoke_logit_{prebuild,release}_graph_nogreedy.json`
- `raw/logit_{prebuild,release}_graph_nogreedy/`
- `raw/logit_parity_graph_nogreedy_rank0.json`
- `raw/moe_micro_parity.json`
- `raw/text_smoke_ab_*.json`
- `raw/marlin_wna16_cache_integrity_*.jsonl`
- `raw/activation_{prebuild,release}_eager_mt4/`
- `raw/activation_divergence_eager_mt4_rank0.json`

## Attribution

Ruled out:

- **Marlin WNA16 prebuild itself**: prebuild-only text smoke passes and matches baseline generated tokens.
- **prepacked runtime branch alone**: `force-prepacked-with-raw-present` passes.
- **normal attribute deletion alone**: `keep-hidden-ref` removes attrs from the normal path but keeps storage alive, and passes.
- **CUDA graph replay as primary cause**: release fails in eager/no-graph too.
- **greedy-sample graph capture as primary cause**: graph/no-greedy still diverges at logits.
- **Marlin packed cache overwrite**: sampled cache `data_ptr`/shape/checksum are stable after release, `empty_cache`, capture, and replay.
- **representative MoE Marlin packed-cache output**: micro parity is exact for sampled layers with same cache before/after release.
- **scale storage release**: scales-only passes; weights-only fails.

Most likely owner:

Physical early release of the large routed expert **weight** storages creates an unsafe lifetime/allocator-reuse condition for full-model decode. The failure needs enough storage to be returned to the allocator and reused by later KV/graph/warmup allocations. Keeping hidden refs, keeping raw attrs, or delaying release until after capture all make text pass.

The first recorded full-model symptom is in attention/indexer on layer2 in an eager activation trace, but that should be treated as the first observed consumer of corrupted state, not a final proof that indexer is the root bug.

## Implementation Notes

Diagnostic-only changes were added behind explicit env flags and scripts:

- `python/minisgl/utils/dsv4_memory_debug.py`
  - cache integrity summaries with `data_ptr`, finite ratio, checksum, shape/dtype/stride.
- `python/minisgl/models/deepseek_v4.py`
  - hidden-ref release A/B;
  - force-prepacked-with-raw-present A/B;
  - partial layer release;
  - weights-only / scales-only release;
  - release-after-graph-capture hook;
  - cache integrity audit hooks.
- `python/minisgl/engine/engine.py` and `python/minisgl/engine/graph.py`
  - cache audits after prepare, KV empty_cache, graph capture, and early replay.
- `benchmark/offline/deepseek_v4_text_smoke.py`
  - graph disable override;
  - greedy sample graph capture disable override;
  - fixed variant env application before LLM construction for eager diagnostic runs.
- `benchmark/offline/deepseek_v4_logit_parity_from_prefix_debug.py`
  - compares prebuild vs release full-logit debug traces.
- `benchmark/offline/deepseek_v4_marlin_wna16_release_micro_parity.py`
  - TP8 MoE packed-cache micro parity probe.

These hooks are intentionally diagnostic. They do not promote release and do not add silent fallback to raw grouped/fallback paths.

## Decision

Final recommendation:

- Do **not** promote `dsv4_sm80_a100_victory_marlin_release`.
- Do **not** promote prefix release presets.
- Keep release experimental/blocked until the storage lifetime owner is fixed.
- Keep `dsv4_sm80_a100_victory_marlin_prebuild` as the safe lifecycle optimization path.
- Do not publish macro throughput for release while this text sanity blocker remains.

The only passing release-style A/B that frees the full `17.1328 GiB/rank` is release-after-capture, but that does not provide the original intended pre-KV/pre-capture capacity headroom. Treat it as attribution evidence, not a promotion-ready fix.

## Validation

Commands are recorded in `COMMANDS.md`.

Runtime gates run for this target:

- three-way 08.35 smoke reuse;
- release graph/eager max-token split;
- graph with greedy-sample capture disabled;
- logit parity ladder;
- MoE packed-cache micro parity;
- release lifetime A/B;
- cache integrity audit;
- eager activation divergence trace.

Macro gates were intentionally not run because release text sanity is still corrupt.

Static validation:

```text
python -m py_compile python/minisgl/utils/dsv4_memory_debug.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/graph.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  benchmark/offline/deepseek_v4_logit_parity_from_prefix_debug.py \
  benchmark/offline/deepseek_v4_marlin_wna16_release_micro_parity.py

pytest -q tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

Result: 76 tests passed. Only third-party deprecation warnings were emitted.

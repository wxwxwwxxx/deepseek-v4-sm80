你好，请帮我在这个项目中调研并实现 DeepSeek-V4-Flash 在
mini-sglang 中的高性能推理，重点是 A100/sm80 适配。

## Project Context

- Framework: `/workspace/mini-sglang`
- Model: `/models/DeepSeek-V4-Flash`
- Official/oracle reference: `/models/DeepSeek-V4-Flash/inference`
- SGLang reference: `/workspace/sglang-main`
- vLLM DeepSeek V4 reference: `/workspace/vllm-dsv4-docker`
- vLLM runtime venv: `/workspace/venvs/vllm-dsv4`
- mini runtime: system Python from `/workspace/mini-sglang`
- Old abandoned mini branch: `dsv4`
- Current main route: use SGLang/vLLM design as high-performance references,
  adapt the parts that are valid on sm80, and avoid re-implementing slow local
  variants when a proven backend can be ported cleanly.

## Global Principles

- Keep the default path exact unless a dedicated precision target proves and
  accepts a quality tradeoff.
- Use page size `256` for DSV4 benchmark and smoke work unless a target says
  otherwise.
- Compare against vLLM/SGLang source behavior before writing a local
  replacement for a major runtime boundary.
- Do not reinvent runtime mechanisms when SGLang/vLLM already has a mature
  design; first map the source behavior, then adapt or port the proven part
  when it fits mini-sglang's constraints.
- Use fair TP8 macro runs, source parity, and focused microbench evidence before
  promoting optimizations.
- Keep large profiler outputs and raw benchmark data under
  `performance_milestones/`; symlink large files when appropriate.
- Archive completed fine-grained prompts so new Codex threads can use the
  current route files instead of replaying the full history.

## Stage Matrix

| Stage | Prompt | Status | Summary |
| --- | --- | --- | --- |
| TARGET 01 | `prompts/TARGET_01_config_registry_weight.md` | completed | DSV4 config/registry/weight-loading groundwork. |
| TARGET 02 | `prompts/TARGET_02_model_forward_fallback.md` | completed | Basic model forward/fallback path. |
| TARGET 03 | `prompts/TARGET_03_dsv4_kvcache_no_radix.md` | completed | DSV4 KV/cache pool without radix prefix cache. |
| TARGET 04 | `prompts/TARGET_04_attention_backend_metadata.md` | completed | DSV4 attention metadata/backend integration. |
| TARGET 05.5 | `prompts/TARGET_05.5_dsv4_sm80_kernel_rd.md` | completed history | Initial sm80 kernel R&D matrix and operator replacement plan. |
| TARGET 05.6 | `prompts/TARGET_05.6_hard_kernel_plans/` | completed history | Early hard-kernel plan set; use as historical reference only. |
| TARGET 05.7 | `prompts/TARGET_05.7_dsv4_v0_bf16_e2e_smoke.md` | completed | Added v0 BF16 E2E smoke and basic correctness gates. |
| TARGET 06 | `prompts/TARGET_06_benchmark_sm80_baseline.md` | completed | Added TP8 benchmark harness and text smoke; fixed early correctness issues. |
| TARGET 07 | `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` | closed | Beat the old vLLM serving line with `dsv4_sm80_a100_victory`; detailed prompts archived under `prompts/archive/target07/`. |
| TARGET 08 | `prompts/TARGET_08_radix_prefix_dsv4.md` | closed prefix baseline plus active SWA capacity/perf children | Built DSV4 radix prefix cache and promoted `dsv4_sm80_a100_victory_prefix_routeb_lifetime`; 08.54 removed the primary SWA direct replay metadata gap with `replaymetafused`, and the next step is TARGET 08.55 residual C4/C128 compressed metadata cleanup plus stop-line decision before TARGET 09. |
| TARGET 09 | `prompts/TARGET_09_dsv4_sm80_low_precision_research.md` | active research | Low-precision research after TARGET 10: INT8 MoE W8A8 and FP8 KV/cache are the two primary lanes; TARGET 09.5 is deferred until TARGET 08.31 proves real SWA lifecycle/capacity value. |
| TARGET 10 | `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` | closed communication baseline | Default-promoted PyNCCL threshold32m for the A100/sm80 DSV4 communication path; detailed prompts archived under `prompts/archive/target10/`. |

## Current Milestones

TARGET 07 non-prefix baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
--page-size 256
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

TARGET 08 prefix-cache baseline:

```text
dsv4-sm80-prefix-routeb-lifetime-baseline
dsv4_sm80_a100_victory_prefix_routeb_lifetime
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

TARGET 08.30 result:

- text smoke/verifier passed;
- graph replay stayed zero-eager;
- `prefix_multi_112req_wave16` improved from `51.0507` to `110.1417` output
  tok/s and saved `49152` prefill tokens;
- no-hit `4096/1024/bs4` stayed close to TARGET 07 control:
  `137.1625` versus `139.8415` output tok/s;
- no-hit `serving_mixed_112req_wave16` still paid opt-in overhead:
  `163.3985` versus `178.3004` output tok/s.

TARGET 10.27 result:

```text
TARGET 10.27 default-promoted PyNCCL threshold32m for the A100/sm80 DSV4 path:
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL enabled by default for that preset
Default DSV4 sm80 PyNCCL max buffer size: 32M unless
MINISGL_PYNCCL_MAX_BUFFER_SIZE is explicitly set.
```

Rollback:

```bash
MINISGL_PYNCCL_MAX_BUFFER_SIZE=1G
# or pass --disable-pynccl on serving / omit PyNCCL benchmark presets
```

Rationale: prefix metadata/runtime is no longer the first bottleneck.  TARGET
10.1 found matching communication owner boundaries and a MoE reduce-once
fp32-vs-BF16 mismatch. TARGET 10.15 fixed that dtype/bytes mismatch as an
explicit BF16 reduce path. TARGET 10.25 and 10.26 showed PyNCCL threshold32m is
repeat-stable positive with zero-eager graph replay. TARGET 10.27 explained the
`lm_head_all_gather` timing spike as a one-time non-captured first all-gather
cost, not a hot-path regression, and captured rank-scoped full-model Nsight
traces with CUDA kernel/memcpy/NCCL/graph activity.

TARGET 09.45 result:

```text
Do not run TARGET 09.5 yet.  Run
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md first.
```

Rationale: current mini keeps SWA as a full 43-layer, 128-page BF16 pool, so
SWA-only FP8 appears to save about `0.576 GiB/rank`.  A SGLang-aligned
independent SWA lifecycle may reduce real long-prefix/prefix-wave SWA retention
to about `4` to `16` tail pages, making SWA-only FP8 much smaller
(`0.018` to `0.072 GiB/rank`) while BF16 lifecycle itself can recover around
`1 GiB/rank` of persistent headroom.  Prove that lifecycle and its runtime
counters before reopening FP8 KV/cache E2E.

TARGET 08.31 result:

```text
prompts/TARGET_08.31_dsv4_sm80_swa_independent_lifecycle.md completed.
Run prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md next.
```

Rationale: SWA independent lifecycle now exists and passes correctness tests.
It keeps C4/C128/indexer/state/component locs valid while SWA tail pages are
tombstoned/freed, and it remains compatible with TARGET 08.40 Marlin WNA16
release + component-slot clear.  Runtime counters prove small live SWA tails,
and Marlin release auto-capacity improved from `2776` to `6636` pages at
roughly the same KV memory budget.  However, fixed-128 serving still uses a
conservative 128-page SWA floor, and short offline E2E output throughput
regressed by about `3%` to `9%`.  TARGET 08.41 should run graph-bucket
`[1,2,4,8,16]` soak, serving/prefix pressure, overhead attribution, and a
promotion/opt-in decision before reopening TARGET 09.5.

TARGET 08.41 result:

```text
prompts/TARGET_08.41_dsv4_sm80_swa_independent_lifecycle_promotion_soak.md completed.
Run prompts/TARGET_08.42_dsv4_sm80_swa_large_capacity_serving_correctness.md next.
```

Rationale: fixed-128 SWA independent serving/prefix/eviction is clean with
graph buckets `[1,2,4,8,16]`, but high-capacity Marlin release + SWA
independent serving crashes with CUDA illegal memory access at auto capacity
and at explicit `--num-pages 4096`.  The stack surfaces near
`_swa_page_refcount`, but likely observes an earlier illegal device access in
large-capacity SWA mapping/page-table/refcount/free-list behavior.  E2E
overhead was attributed mainly to decode prepare / attention metadata, but
correctness must be fixed before overhead optimization or TARGET 09.5 FP8 work.
TARGET 08.42 should investigate no-weight/partial repros first, then confirm
with full-model smoke/macro.

TARGET 08.42 result:

```text
prompts/TARGET_08.42_dsv4_sm80_swa_large_capacity_serving_correctness.md completed.
Run prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md next.
```

Rationale: the large-capacity Marlin release + SWA independent crash was caused
by an Engine/KV-cache dummy full-page token contract mismatch.  Engine planned
`num_tokens = planned_pages * page_size`, allocated `planned_pages + 1` pages
for the dummy request row, and used `num_tokens` as the dummy full-token start;
DSV4 KV-cache previously treated `allocated_pages * page_size` as the dummy
sentinel.  That made graph-padded dummy rows look like real full pages and
translate to SWA page `-1`, corrupting device memory.  The fix passes
`dsv4_dummy_token_start=num_tokens` into `DeepSeekV4KVCache` and maps the
dummy row to the SWA dummy page.  No-weight repros verified the producer, and
full-model text/serving confirmation passed fixed 128, explicit cap4096, and
auto-capacity paths with zero eager replay in the tested serving macro.  TARGET
08.43 should now rerun the promotion soak, quantify remaining overhead, and
decide promote vs opt-in.

TARGET 08.43 result:

```text
prompts/TARGET_08.43_dsv4_sm80_swa_independent_post_fix_promotion_soak.md completed.
Run prompts/TARGET_08.44_dsv4_sm80_swa_stale_prefix_handle_tombstone_fix.md next.
```

Rationale: TARGET 08.43 confirmed the 08.42 dummy-token fix remained healthy,
but SWA independent lifecycle still cannot be promoted.  Fixed128 long decode
`historical_4096_1024_bs4` failed on all ranks with
`DSV4 KV cache double free detected in SWA page slots`; graph replay was
healthy before failure (`1277` replay, `0` eager), and the pure SWA independent
variant failed before Marlin release + SWA independent.  The likely owner is
stale prefix SWA handles: active decode out-of-window release frees a SWA page,
then finish-time prefix tombstone revisits an older handle containing the same
page.  TARGET 08.44 should preserve the no-weight repro as a regression test,
fix active-release / prefix-handle tombstone synchronization, and pass the
fixed128 `4096/1024/bs4` SWA gates before rerunning 08.43.

TARGET 08.44 result:

```text
prompts/TARGET_08.44_dsv4_sm80_swa_stale_prefix_handle_tombstone_fix.md completed.
Run prompts/TARGET_08.45_dsv4_sm80_swa_independent_lifecycle_contract.md next.
```

Rationale: TARGET 08.44 reproduced and fixed the no-weight/core stale
prefix-handle double-free path and kept the true duplicate-release guard
active.  Focused tests passed (`110 passed`).  However, the TP8 full-model
fixed128 gate then hit CUDA illegal memory access in the first SWA independent
case, before the `historical_4096_128_bs4` report was emitted.  This suggests
the SWA independent lifecycle still lacks a fully closed ownership/metadata
contract.  Do not continue with ad hoc single-point patches.  TARGET 08.45
should write `prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md` aligned with
SGLang, TARGET 08.46 should audit mini implementation against that contract,
and TARGET 08.47 should apply a unified fix before rerunning TARGET 08.43
promotion soak.

TARGET 08.32-08.40 CUDA graph / warmup memory follow-up:

```text
prompts/TARGET_08.32_dsv4_sm80_cuda_graph_private_pool_micro_attribution.md
prompts/TARGET_08.33_dsv4_sm80_indexer_capture_static_width_audit.md
prompts/TARGET_08.34_dsv4_sm80_moe_marlin_wna16_cache_lifecycle.md
prompts/TARGET_08.35_dsv4_sm80_marlin_wna16_release_preset_promotion.md
prompts/TARGET_08.36_dsv4_sm80_marlin_wna16_release_correctness_attribution.md
prompts/TARGET_08.37_dsv4_sm80_marlin_wna16_release_storage_reuse_owner.md
prompts/TARGET_08.38_dsv4_sm80_marlin_wna16_safe_release_arena_capacity.md
prompts/TARGET_08.39_dsv4_sm80_marlin_wna16_old_address_root_cause.md
prompts/TARGET_08.40_dsv4_sm80_marlin_wna16_release_component_clear_promotion.md
```

Rationale: TARGET 08.06/08.07 proved the `~19 GiB/rank` first-graph CUDA graph
private-pool cost is real and not explained by graph input buffers, greedy
sampling, captured compressed-loc metadata, `max_seq_len`, `num_pages`, bucket
count, or BF16 projection/shared-expert caches.  This is likely one of the
largest remaining capacity/headroom opportunities.  TARGET 08.32 ruled out many
synthetic owners but did not find the full-model source.  TARGET 08.33 falsified
the DSV4 C4 indexer-width hypothesis and showed the large movement happens
during warmup `model.forward()`, before the actual `torch.cuda.graph` block.
TARGET 08.34 is the next focused investigation: audit whether the default
`marlin_wna16` MoE backend lazily repacks and retains routed expert weights on
first forward, creating roughly `17-18 GiB/rank` of persistent backend state
that should be prebuilt/accounted before KV capacity planning.

TARGET 08.34 result: the hypothesis was confirmed.  Prebuild removes the
warmup-forward spike by moving Marlin WNA16 cache creation before KV capacity
planning, and releasing original routed FP4 expert weights recovers about
`17.13 GiB/rank`, around `400` KV pages or `102k` tokens per rank at page size
`256`.  TARGET 08.35 should make prebuild+release a named memory-efficient
preset rather than a loose opt-in, with correctness, graph replay, macro, and
fail-closed backend gates.

TARGET 08.35 result: naming, env expansion, two-stage prebuild/release, memory
ledger, and fail-closed semantics landed, but release promotion was rejected.
Baseline and prebuild-only text smoke passed; the release preset recovered
`17.1328 GiB/rank` but produced corrupted text.  TARGET 08.36 should attribute
that correctness blocker before any release preset is promoted.

TARGET 08.36 result: release remains blocked, but the evidence supports
continuing the release route.  Eager/no-graph release also fails, so CUDA graph
replay is not the primary cause.  `force-prepacked-with-raw-present`,
`keep-hidden-ref`, and `release-after-capture` pass, while `weights-only`
release fails and `scales-only` passes.  The failure follows early physical
release of large expert-weight storages and likely allocator reuse by a later
KV/cache/warmup/graph/attention/indexer owner.  TARGET 08.37 should identify
that owner or the earliest safe release boundary.

TARGET 08.37 result: the unsafe owner was identified as the DSV4 KV/component
allocation phase after immediate release.  KV/component tensors reused released
raw expert-weight ranges; releasing after KV allocation passes but does not
provide pre-KV capacity-planning headroom.  TARGET 08.38 should repair this by
planning with a release credit while using an arena/guard/allocation-order
policy to keep live KV/component buffers off unsafe released ranges.

TARGET 08.38 result: a `before_kv_alloc` release with a `3.1875 GiB/rank`
deterministic guard arena passed short text smokes, graph replay, and the
historical 4096x128 / 4096x1024 macro shapes.  Auto-planned capacity improved
from `1,826` to `2,602` pages at page size `256`.  The guard recovered most of
the `17.1328 GiB/rank` raw-expert release value, but it is still empirical: in
rank-0 records it maps to the first 32 released items, layers `0-7` with
`w13_weight`, `w13_weight_scale_inv`, `w2_weight`, and
`w2_weight_scale_inv`.  TARGET 08.39 should now chase the root cause with old
expert address NaN/byte poison, KV-as-sentinel probes, stage/layer bisection,
and stream-lifetime controls, aiming for unguarded release where KV/component
can safely use the formerly raw expert-weight ranges.

TARGET 08.39 result: the bug was attributed to an uninitialized DSV4 component
cache read after allocator reuse of old raw expert-weight storage.  The
effective fix is component-slot clear on page allocation: `clear=component`
passes, while `clear=none`, `clear=full`, and `clear=state` fail or warn.
`CUDA_LAUNCH_BLOCKING=1 + clear=none` still fails, making stream lifetime
unlikely as the root.  Fixed unguarded release passes eager/graph text smoke,
uses `0` guard bytes, still lets KV/component overlap old raw expert ranges,
and auto-plans `2,779` pages at page size `256`.  TARGET 08.40 should now
productionize the fix, add regression coverage, measure page-allocation clear
overhead, run macro/prefix/serving gates, and decide whether to promote the
Marlin WNA16 release preset.

## Archive Policy

Completed detailed execution prompts live in:

```text
prompts/archive/target07/
prompts/archive/target08/
prompts/archive/target10/
```

For new child threads, start from:

1. `prompts/target.md`
2. the active target prompt, currently TARGET 08.55 or a TARGET 09 child
3. `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` only for TARGET 07
   milestone history
4. `prompts/TARGET_08_radix_prefix_dsv4.md` for prefix-cache history and
   deferred cache work
5. `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` for the
   closed communication default and rollback policy

Do not ask new threads to read every archived prompt unless they need exact
historical commands or stop conditions.

## Long-Term Cache / Workspace Principle

As DSV4 stabilizes, converge persistent cache, temporary workspace,
pre-dequantized weights, CUDA graph capture buffers, and low-precision cache
state into clear management entry points.

The desired direction is:

- capacity planning before model prepare and graph capture;
- no repeated large `cudaMalloc` or hidden rebuild during decode graph replay;
- every cache/workspace reports owner, shape, dtype, bytes, lifecycle, and
  equivalent KV-token cost;
- local optimization experiments may start as opt-ins, but promoted paths should
  be auditable through unified cache/workspace ownership.

## Useful Commands

TARGET 06 baseline example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback v0_bf16 \
  --page-size 256 \
  --output-dir /tmp/dsv4_sm80_target06_tp8 \
  --keep-going
```

TARGET 08 prefix baseline example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --keep-going
```

TARGET 08 text correctness smoke example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --verify-dsv4-route-b-cache \
  --output /tmp/dsv4_prefix_text_smoke.json
```

## Release-Style Serving Benchmark Direction

Before declaring the serving path broadly usable, run a more complete serving
benchmark pass:

- `requests >= 100` when runtime allows;
- multiple request-rate or arrival-pattern settings, for example RPS
  `0.2, 0.5, 1, 2, 4, 8`;
- fixed max concurrency settings;
- short-output and long-output workloads;
- shared-prefix and non-shared-prefix mixes;
- GPU utilization;
- KV cache usage;
- active batch-size distribution;
- queueing latency, TTFT, ITL/TPOT, and output throughput.

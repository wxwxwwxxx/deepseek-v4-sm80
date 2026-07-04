你好，请帮我在这个项目中调研并实现 DeepSeek-V4-Flash 在
mini-sglang 中的高性能推理，重点是 A100/sm80 适配。

## Project Context

- Framework: `/workspace/mini-sglang`
- Model: `/models/DeepSeek-V4-Flash`
- Official/oracle reference: `/models/DeepSeek-V4-Flash/inference`
- SGLang reference: `/workspace/sglang-main`
- vLLM DeepSeek V4 reference: `/workspace/vllm-dsv4-docker`
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
| TARGET 08 | `prompts/TARGET_08_radix_prefix_dsv4.md` | active family | Conservative DSV4 radix prefix cache exists as explicit opt-in; continue with TARGET 08 subtargets. |
| TARGET 08.05 | `prompts/TARGET_08.05_dsv4_sm80_serving_workload_cuda_graph_bucket_policy.md` | completed | Established serving workload suite and selected `[1,2,4,8,16]` as the smallest measured zero-eager bucket set. |
| TARGET 08.06 | `prompts/TARGET_08.06_dsv4_sm80_cuda_graph_memory_attribution.md` | completed | Confirmed the large capture delta is a real first-graph/private-pool cost, not bucket count, metadata, greedy sample, `max_seq_len`, `num_pages`, or missing pool reuse. |
| TARGET 08.07 | `prompts/TARGET_08.07_dsv4_sm80_bf16_cache_graph_memory_attribution.md` | completed | Ruled out promoted BF16 caches as the material cause of the large CUDA graph private-pool delta. |
| TARGET 08.10 | `prompts/TARGET_08.10_dsv4_sm80_prefix_cache_serving_stability_promotion_gate.md` | completed controlled opt-in | Validated prefix cache under serving-like sustained workloads, but kept it opt-in because generated-token correctness was not yet a clean promotion oracle. |
| TARGET 08.18 | `prompts/TARGET_08.18_dsv4_sm80_prefix_cache_memory_ledger_go_nogo.md` | completed | Computed full-page-owner prefix-cache memory/capacity cost and recommended guarded component-retention work. |
| TARGET 08.19 | `prompts/TARGET_08.19_dsv4_sm80_prefix_cache_logit_metadata_correctness.md` | completed blocked | Prefix metadata was clean, but logits exposed a DSV4 exact-path slot/page-location blocker. |
| TARGET 08.195 | `prompts/TARGET_08.195_dsv4_sm80_exact_path_slot_page_invariance.md` | completed partial fix | Fixed a real compressor cross-request pooling bug and established guards, but remaining batched attention/indexer row-coupling still blocks broad oracle use. |
| TARGET 08.196 | `prompts/TARGET_08.196_dsv4_sm80_batched_attention_indexer_row_coupling.md` | completed narrowed | Added attention/indexer debug hooks and exact-bs graph guard; found layer0 q-path drift but did not clear broad correctness. |
| TARGET 08.197 | `performance_milestones/target08_q_path_same_shape_same_input_invariance/README.md` | completed classification | Classified layer0 q-path drift as GEMM shape numeric drift, not q_norm/RoPE row-coupling. |
| TARGET 08.198 | `prompts/TARGET_08.198_dsv4_sm80_post_layer0_same_shape_decode_drift.md` | completed guarded | Found tiny later-layer attention/indexer drift amplified by small logits margins; accepted guarded oracle because batch-slot invariance is not guaranteed. |
| TARGET 08.20 | `prompts/TARGET_08.20_dsv4_sm80_sglang_style_swa_component_retention.md` | completed rejected | Added fail-closed V1 opt-in and proved runtime V1 is unsafe without component-level ownership. |
| TARGET 08.21 | `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md` | route overview | Route B family map; do not run as one monolithic implementation target. |
| TARGET 08.21.1 | `prompts/TARGET_08.21.1_dsv4_sm80_component_loc_table_preflight.md` | completed | B0: proved direct component loc tables match phase-1 derived metadata while full pages stay live. |
| TARGET 08.21.2 | `prompts/TARGET_08.21.2_dsv4_sm80_independent_compressed_indexer_ownership.md` | completed | B1: independent C4/C128/indexer ownership behind an opt-in. |
| TARGET 08.21.3 | `prompts/TARGET_08.21.3_dsv4_sm80_compression_state_ownership.md` | completed | B2: independent C4/C128/indexer compression-state ownership; SWA-tail guard remains. |
| TARGET 08.21.4 | `prompts/TARGET_08.21.4_dsv4_sm80_route_b_graph_deforest_serving.md` | completed preferred opt-in candidate | B3: Route B graph metadata/copy restored for `[1,2,4,8,16]`; deforest guarded; full gate needed. |
| TARGET 08.22 | `prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md` | completed preferred opt-in | Final Route B rerun passed correctness/text/graph and selected Route B as the preferred prefix-cache opt-in. |
| TARGET 08.22.1 | `prompts/TARGET_08.22.1_dsv4_sm80_route_b_component_mapping_lifecycle_fix.md` | completed | Fixed Route B active full-page to component-page mapping lifecycle for multi-page serving reuse. |
| TARGET 08.23 | `prompts/TARGET_08.23_dsv4_sm80_independent_swa_ownership.md` | deferred conditional | SGLang-aligned independent SWA ownership only if later evidence shows the SWA-tail guard materially blocks serving capacity or hit rate. |
| TARGET 08.24 | `prompts/TARGET_08.24_dsv4_sm80_route_b_metadata_deforest_copy_elision.md` | completed keep experimental | Component-aware metadata generation was correct but slower because it still materialized and staged large source tensors. |
| TARGET 08.25 | `prompts/TARGET_08.25_dsv4_sm80_route_b_direct_graph_metadata_buffers.md` | completed keep experimental | Direct graph metadata buffers were safe, but large-wave gains were too small and full direct generation was not promotable. |
| TARGET 08.26 | `prompts/TARGET_08.26_dsv4_sm80_route_b_remaining_gap_attribution_reset.md` | completed | Re-ranked the remaining Route B gap to decode-prepare component page-table/metadata lifetime overhead. |
| TARGET 08.27 | `prompts/TARGET_08.27_dsv4_sm80_sglang_aligned_route_b_metadata_lifetime.md` | completed strong opt-in | Added SGLang-aligned Route B component page-table lifetime cache; serving mixed improved to `162.47` output tok/s. |
| TARGET 08.28 | `prompts/TARGET_08.28_dsv4_sm80_route_b_lifetime_cache_promotion_gate.md` | completed promote | Promoted the 08.27 lifetime cache after verifier/text/eviction/prefix_multi/decode-control gates. |
| TARGET 08.29 | `prompts/TARGET_08.29_dsv4_sm80_route_b_lifetime_promotion_cleanup.md` | completed cleanup | Created promoted preset `dsv4_sm80_a100_victory_prefix_routeb_lifetime` and kept the old Route B lifetime diagnostic name as an alias. |
| TARGET 08.30 | `prompts/TARGET_08.30_dsv4_sm80_post_prefix_reprofile_next_bottleneck.md` | active next | Reprofile the promoted Route B lifetime prefix preset, then choose TARGET 09 low precision, TARGET 10 attention/communication, more TARGET 08 cache work, or serving hardening. |
| TARGET 09 | `prompts/TARGET_09_dsv4_sm80_low_precision_research.md` | planned after TARGET 08 | Low-precision research: FP8 KV/cache/indexer, INT8 MoE, quantized projection/cache fusion. |
| TARGET 10 | `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` | future optional | Attention, PyNCCL, communication overlap, and graph/runtime experiments if fresh profiles justify them. |

## Current Milestone

TARGET 07 final promoted path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Post-07.78 stable retest:

- 4096/1024/batch4: `131.7561 output tok/s` mean;
- 4096/128/batch4: `62.3925 output tok/s` mean;
- graph replay active;
- eager decode `0`;
- old serving baseline crossed: `114.07 output tok/s`.

Decision from TARGET 07.79 through TARGET 08.29:

```text
continue with TARGET 08.30 DSV4 post-prefix reprofile
```

Reason: DSV4 radix prefix cache works as an explicit opt-in, and Route B is now
the preferred prefix-cache ownership route after TARGET 08.22 rerun.  Route B
recovers most phase-1 saved-prefill tokens and preserves graph replay, but it
still pays a large decode-prepare tax.  TARGET 08.24 and TARGET 08.25 proved
component-aware metadata generation and direct graph metadata buffers are safe
but not yet fast enough.  TARGET 08.26 reset attribution and found the remaining
gap is dominated by component page-table and metadata lifetime work repeated
across decode replay steps.  TARGET 08.27 addressed that owner with a
SGLang-aligned request/table-slot keyed component page-table lifetime cache:
`serving_mixed_112req_wave16` improved from `138.1281` to `162.4726` output
tok/s, decode prepare dropped from `4.2067 s` to `1.1416 s`, and graph replay
remained `441/0`.  TARGET 08.28 then promoted this path after verifier, text
smoke, prefix_multi, eviction pressure, decode-control, and graph replay gates.
`serving_mixed_112req_wave16` reached `163.7220` output tok/s with `441/0`
replay.  TARGET 08.29 then cleaned this up into the promoted benchmark/text
smoke preset `dsv4_sm80_a100_victory_prefix_routeb_lifetime`, preserving
`MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1` across variant
env reset and keeping
`dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime` as a historical
alias.

The next target should not invent a new runtime mechanism.  TARGET 08.30 should
run the global post-prefix bottleneck reset against
`dsv4_sm80_a100_victory_prefix_routeb_lifetime`.

## Archive Policy

Completed TARGET 07 execution prompts live in:

```text
prompts/archive/target07/
```

For new child threads, start from:

1. `prompts/target.md`
2. the active target prompt, currently
   `prompts/TARGET_08.30_dsv4_sm80_post_prefix_reprofile_next_bottleneck.md`
3. `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` only for milestone history
4. `prompts/TARGET_08_radix_prefix_dsv4.md` for prefix-cache phase-1 context

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

TARGET 06 text correctness smoke example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants fallback v0_bf16 \
  --output /tmp/dsv4_text_smoke.json
```

## Release-Style Serving Benchmark Direction

Before declaring the serving path broadly usable, run a more complete serving
benchmark pass.  TARGET 08.05 and TARGET 08.30 should use this as guidance:

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

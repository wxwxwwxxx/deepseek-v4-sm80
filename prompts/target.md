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
| TARGET 08 | `prompts/TARGET_08_radix_prefix_dsv4.md` | closed prefix baseline plus SWA/metadata history | Built DSV4 radix prefix cache, Route-B ownership, SWA lifecycle work, and direct replay metadata cleanup; detailed child targets remain as history for prefix/SWA correctness and capacity work. |
| TARGET 09 | `prompts/TARGET_09_dsv4_sm80_low_precision_research.md` | deferred | Low-precision research is paused after the INT8 MoE feasibility pass did not show an obvious short win; keep the evidence for later INT8/FP8 work. |
| TARGET 10 | `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` | closed communication baseline | Default-promoted PyNCCL threshold32m for the A100/sm80 DSV4 communication path; detailed prompts archived under `prompts/archive/target10/`. |
| TARGET 11 | `prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md` | paused and archived | MTP speculative decoding was investigated and preserved on `dsv4-mtp-paused-reference`, but the current target-verify runtime failed the no-spec target decode equivalence contract.  Current release branch removes active MTP runtime/opt-ins and should establish a post-MTP-cleanup non-MTP baseline. |
| TARGET 12 | `prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md` | active fallback census | Post-MTP-cleanup non-MTP performance follow-up: promoted the DSV4 A100/sm80 release bundle with SWA independent lifecycle, HC prenorm temporary elimination, and conservative 8192-token chunked prefill; current work is fallback/native-backend census after 12.56 showed larger chunks are limited by indexer fallback temporaries. |

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
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
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

TARGET 08/09 follow-up summary:

```text
TARGET 08 child prompts are archived under prompts/archive/target08/.
TARGET 09 child prompts are archived under prompts/archive/target09/.
```

TARGET 08 post-prefix work is now summarized in `prompts/TARGET_08_radix_prefix_dsv4.md`:

- TARGET 08.31-08.48 implemented and contract-audited SWA independent lifecycle, fixed large-capacity dummy-token mapping, stale prefix-handle tombstones, and same-Engine Marlin release + SWA address issues.
- TARGET 08.34-08.40 identified Marlin WNA16 lazy cache creation as the large warmup/capacity owner, then made original routed expert weight release safe by clearing component slots on page allocation.  This recovers about `17 GiB/rank` of raw expert storage for KV/component capacity.
- TARGET 08.49-08.55 reduced SWA/prefix metadata overhead through page-table caching, direct token metadata, graph/copy attribution, and direct replay metadata fusion.  The remaining metadata kernels are no longer worth polishing without a fresh profile showing them as top bottlenecks.
- Prefix cache and SWA lifecycle should be treated as important serving/capacity baselines, but TARGET 08 itself is closed unless a future feature changes prefix/SWA/cache ownership.

TARGET 09 low-precision work is now summarized in `prompts/TARGET_09_dsv4_sm80_low_precision_research.md`:

- INT8 MoE remains a possible research lane, but the feasibility pass did not yet identify a low-risk W8A8 backend/quantization path that clearly beats the current MXFP4/WNA16 Marlin expert route.
- FP8 KV/cache should not proceed as a broad E2E feature until a fresh memory ledger shows real ROI after SWA lifecycle and Marlin release capacity improvements.
- Dense FP8 projection is currently a memory/capacity feature, not a throughput win.
- TARGET 09 is deferred until a fresh profile or memory ledger makes
  low-precision work clearly valuable again.
- TARGET 11 MTP is paused for release; its code/debug history is preserved on
  `dsv4-mtp-paused-reference`, and fine-grained prompts are archived under
  `prompts/archive/target11/`.

TARGET 12 starting point:

```text
TARGET 12 follows the post-MTP-cleanup replay attribution report:
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md

Current evidence says the remaining non-MTP regression is concentrated around
decode CUDA graph replay setup and metadata preparation/staging before
`g.replay()`, not communication count/bytes, wrapper count, or graph replay
coverage.
```

TARGET 12 should first compare mini's replay boundary with SGLang and vLLM:

- SGLang's decode CUDA graph runner uses stable input buffers and grouped GPU
  copies, and its DeepSeek V4 backend can convert raw decode metadata into full
  graph-consumed metadata inside the captured graph.
- vLLM's CUDA graph wrapper keeps graph replay separate from persistent buffer
  ownership, and its DeepSeek V4 attention path uses preallocated metadata
  surfaces around sparse/SWA attention.
- Multi-stream overlap is not part of the current TARGET 12 route: mini's
  measured replay metadata byte volume is small, and same-step metadata still
  feeds `g.replay()`, so stable buffers, deforestation, in-graph prep, and
  direct/fused graph metadata writers have higher priority.

Current TARGET 12 child:

```text
prompts/TARGET_12.57_dsv4_sm80_release_fallback_census_native_backend_gate.md
```

TARGET 12.4 implemented an opt-in SGLang-style in-graph metadata prep PoC with
the current replay metadata path preserved as fallback/oracle.  It removed the
main `prepare_for_replay` clamp/copy owner in the short probe and passed
unit/wrapper/text gates without increasing capture memory.  TARGET 12.45 then
showed repeat-stable positive performance across short, long-decode, serving,
and prefix scenarios.  TARGET 12.46 fixed the long-context
`c4_sparse_raw_indices` oracle boundary around indexer-mutated C4 fields.
TARGET 12.47 reran the promotion subset with one fresh process per variant and
confirmed correctness plus repeat-stable macro wins. TARGET 12.48 folds that
recipe into the DSV4 A100/sm80 release defaults.

Current release-default intent after TARGET 12.53:

```text
LLM("/models/DeepSeek-V4-Flash", ...)

DeepSeek V4 Engine defaults:
page_size=256 when the config still has the generic default page_size=1
attention_backend=dsv4
radix prefix cache enabled
component loc ownership enabled
cuda_graph_bs=[1,2,4,8,16]

Release env defaults when no explicit MINISGL_DSV4_SM80_* runtime env is set:
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_LINEAR_BF16_FP32=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
```

Fallback/oracle paths remain available through benchmark variants or:

```bash
MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1
```

The first TARGET 12.49 attempt showed that the release-default recipe was
missing Marlin WNA16 prebuild/release/capacity-credit behavior before automatic
KV planning. TARGET 12.50 fixed that and promoted the Tier A release bundle.  The true
no-env release-default smoke now passes with text sanity, CUDA graph capture
for `[1,2,4,8,16]`, Marlin WNA16 prebuild/release capacity credit, and
automatic KV planning.

TARGET 12.50 kept SWA independent lifecycle opt-in even though it was correct
and graph-replay clean, because it still disabled in-graph metadata prep:

```text
prep_metadata_in_graph_requested=true
prep_metadata_in_graph=false
prep_metadata_in_graph_unsupported_reason="swa_independent_lifecycle_not_supported"
```

TARGET 12.51 fixed that blocker.  SWA independent now uses
`prep_metadata_in_graph=true`, passes oracle/text/graph/macro gates, and keeps
the large capacity upside:

```text
Tier A default:       2763 pages / 707,328 tokens
SWA independent path: 6457 pages / 1,652,992 tokens
per-page KV bytes:    19,313,920 B -> 8,041,728 B
```

In the 12.51 paired macro gate, the SWA candidate was positive versus same-run
release default by about `+1.15%` to `+9.52%` output tok/s, with zero eager
decode fallback. TARGET 12.52 then promoted this path into the true no-env
`dsv4_sm80_release_default` bundle. The 12.52 default smoke and macro gates
passed with `prep_metadata_in_graph=true`, zero eager decode fallback, and
about `1.65M` tokens of TP8 A100 planned capacity.

TARGET 12.49 reran long-context and large-batch soak on the 12.52 release
default.  The release path was healthy for text smoke, 8192-token long context,
default-bucket large-batch decode through batch 128, and explicit graph bucket
probes through max bucket 128.  It exposed one high-priority blocker: at 32768
prefill tokens, both `prompt_len=32768,batch=1` and `prompt_len=128,batch=256`
OOM in `hc_pre_fallback` on a 2 GiB FP32 prenorm temporary from
`flat.float().square().mean(...)`.

TARGET 12.53 fixed that blocker and promoted the HC release pair:

```text
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_LINEAR_BF16_FP32=1
```

The 12.53 report shows the old OOM shapes now pass and the four historical
macro rows improve by about 5-9%:

```text
performance_milestones/target12_hc_prenorm_temp_elimination/README.md
```

TARGET 12.54 reran the true no-env release envelope after HC promotion.  Text
sanity passed and `32768/16/1` now passes, but `65536/8/1` OOMs during first
prefill in attention `wo_a` BF16 BMM:

```text
performance_milestones/target12_post_hc_release_envelope_rerun/README.md
```

The failing allocation is only `128 MiB` with about `45 MiB` free, while the
planned KV capacity remains about `1.6M` tokens.  This points to release KV
planning being too close to the memory limit: graph private pool,
activation/workspace peak, fixed SWA cache, and allocator slack are not yet
first-class reserves.

TARGET 12.55 ran the memory-ratio sweep and concluded:

```text
CHUNKED_PREFILL_REQUIRED
performance_milestones/target12_graph_activation_memory_accounting/README.md
```

Lowering `memory_ratio` from `0.90` to `0.85` freed about `3.92 GiB`,
`524 pages`, or `134,144 tokens` of KV capacity, but `65536/8/1` still OOMed.
As memory headroom increased, the failure owner moved from attention `wo_a`
to MoE gate and then Marlin WNA16 MoE `route_out` workspace.  The Marlin owner
is expected for full 64k prefill because `route_out` scales roughly as:

```text
[tokens * topk, hidden] bf16
```

Mini already has a `ChunkedReq` / `PrefillManager.token_budget` skeleton, and
`SchedulerConfig.max_extend_tokens` defaults to `8192`.  However, the offline
perf matrix had been setting `max_extend_tokens` to the scenario's full prefill
length unless `--max-extend-tokens` was explicitly passed, so TARGET 12.54 and
12.55 intentionally tested monolithic prefill.

TARGET 12.56 validated and hardened the existing chunked-prefill path:

```text
performance_milestones/target12_chunked_prefill_long_context/README.md
```

It fixed DSV4 SWA/component capacity accounting for long-request admission,
carried SWA eviction state across `ChunkedReq` segments, preserved decode CUDA
graph replay for captured buckets, and selected `8192` as the conservative
DSV4 A100/sm80 release prefill chunk token budget.  `24576` was fastest at 65k
but failed at 131k; `16384` passed 131k but failed at 262k; `8192` passed 262k
with 32 eager prefill chunks and decode graph replay unchanged.  The next work
is TARGET 12.57, focused on the indexer/fallback temporaries that limit larger
chunk budgets.

## Archive Policy

Completed detailed execution prompts live in:

```text
prompts/archive/target07/
prompts/archive/target08/
prompts/archive/target09/
prompts/archive/target10/
prompts/archive/target11/
```

For new child threads, start from:

1. `prompts/target.md`
2. the current route prompt for the task; after MTP cleanup, TARGET 12 is the
   active non-MTP performance route
3. `prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md` only for the MTP
   pause report and future restart conditions
4. `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` only for TARGET 07
   milestone history
5. `prompts/TARGET_08_radix_prefix_dsv4.md` for prefix-cache history and
   SWA/cache ownership history
6. `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` for the
   closed communication default and rollback policy
7. `prompts/TARGET_09_dsv4_sm80_low_precision_research.md` only when reopening
   deferred low-precision research
8. `prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md` for
   decode replay metadata, graph-buffer, safe replay attribution, in-graph
   metadata prep, and direct/fused graph metadata writer work

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

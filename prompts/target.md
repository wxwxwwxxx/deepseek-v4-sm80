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
| TARGET 08 | `prompts/TARGET_08_radix_prefix_dsv4.md` | closed baseline | Built DSV4 radix prefix cache and promoted `dsv4_sm80_a100_victory_prefix_routeb_lifetime` as the prefix-cache baseline; detailed prompts archived under `prompts/archive/target08/`. |
| TARGET 09 | `prompts/TARGET_09_dsv4_sm80_low_precision_research.md` | planned | Low-precision research: FP8 KV/cache/indexer, INT8 MoE, quantized projection/cache fusion. |
| TARGET 10 | `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md` | recommended family | Post-prefix profiles point to decode-forward communication/all-reduce owners as the next evidence-based surface. |
| TARGET 10.1 | `prompts/TARGET_10.1_dsv4_sm80_comm_path_parity_vllm.md` | completed | Compared mini and vLLM communication owner boundaries; found matching boundaries but a high-severity MoE reduce-once fp32-vs-BF16 dtype/bytes mismatch. |
| TARGET 10.15 | `prompts/TARGET_10.15_dsv4_sm80_moe_reduce_bf16_parity.md` | completed | Implemented BF16 MoE reduce-once as an explicit opt-in; hot fp32 all-reduce disappeared, but promotion awaits repeat-stable evidence. |
| TARGET 10.2 | `prompts/TARGET_10.2_dsv4_sm80_comm_stack_backend_experiments.md` | completed | Tested Torch/NCCL, mini PyNCCL, symmetric-memory workspace, and no-weight replay; best candidate was PyNCCL threshold32m opt-in, not yet promoted. |
| TARGET 10.25 | `prompts/TARGET_10.25_dsv4_sm80_comm_size_owner_routing.md` | completed | Repeat-gated PyNCCL threshold32m as a positive opt-in; explicit owner/size routing did not beat the global threshold cheap gate. |
| TARGET 10.26 | `prompts/TARGET_10.26_dsv4_sm80_pynccl_threshold32m_promotion_gate.md` | completed | PyNCCL threshold32m became the recommended opt-in: repeat-stable macro wins and zero-eager graph replay, but default promotion was blocked by an `lm_head_all_gather` owner-timing anomaly and a full serving Nsight capture without CUDA activity. |
| TARGET 10.27 | `prompts/TARGET_10.27_dsv4_sm80_pynccl_default_promotion_blockers.md` | completed | Resolved the `lm_head_all_gather` owner-timing artifact and full-model Nsight evidence gap; PyNCCL threshold32m is now the default A100/sm80 DSV4 communication path. |

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

## Archive Policy

Completed detailed execution prompts live in:

```text
prompts/archive/target07/
prompts/archive/target08/
```

For new child threads, start from:

1. `prompts/target.md`
2. the active target prompt, usually TARGET 10 or TARGET 09
3. `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md` only for TARGET 07
   milestone history
4. `prompts/TARGET_08_radix_prefix_dsv4.md` for prefix-cache history and
   deferred cache work

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

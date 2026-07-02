# TARGET 07.67: Post-Shared-Expert Reprofile

Date: 2026-07-02

## Summary

TARGET 07.67 was measurement/decision only.  No runtime optimization, precision
route, communication change, attention rewrite, shared-expert overlap, or bundle
promotion was implemented.

The promoted `dsv4_sm80_a100_victory` path reproduces the 07.66 audit variant:

- 4096/128/batch4: `62.1364 output tok/s` vs 07.66 `62.2034` (`-0.11%`).
- 4096/1024/batch4: `131.6263 output tok/s` vs 07.66 `131.7707` (`-0.11%`).
- Graph replay remains active; eager decode remains `0`.
- Fresh direct-copy total is `0.449200s`, matching 07.66 `0.449052s`.
- Shared expert projection owners remain absent.

Recommended next target: **HC / elementwise graph cleanup**.

## Baseline Context

Git context recorded at start:

```text
branch: dsv4-sglang-based
short commit: 9769c9a
status: ahead origin/main by 39 commits, dirty worktree
```

Full `git status --short --branch` snapshot:

```text
## dsv4-sglang-based...origin/main [ahead 39]
M  benchmark/offline/deepseek_v4_perf_matrix.py
M  benchmark/offline/deepseek_v4_text_smoke.py
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/README.md
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/scripts/classify_direct_copy_owners.py
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/scripts/nsys_direct_copy_owner_4096x128_bs4.sh
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/summaries/mini_target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128_nsys_summary.json
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/summaries/nsys_target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128_rank0_direct_copy_owner.json
A  performance_milestones/target07_moe_shared_expert_staging_cleanup/summaries/nsys_target0766_dsv4_sm80_a100_victory_sharedbf16_4096x128_bs4_np128_rank0_direct_copy_owner.md
M  prompts/TARGET_07.66_dsv4_sm80_moe_shared_expert_staging_cleanup.md
A  prompts/TARGET_07.67_dsv4_sm80_post_shared_expert_reprofile.md
M  prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md
M  prompts/target.md
M  python/minisgl/kernel/deepseek_v4.py
M  python/minisgl/models/deepseek_v4.py
M  tests/benchmark/test_deepseek_v4_perf_matrix.py
M  tests/benchmark/test_deepseek_v4_text_smoke.py
M  tests/kernel/test_deepseek_v4_wrappers.py
M  tests/models/test_deepseek_v4_forward_fallback.py
?? performance_milestones/target07_post_shared_expert_reprofile/
```

The dirty worktree already contained TARGET 07.66 source/test/prompt changes.
TARGET 07.67 added only this milestone directory.

Promoted variant:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Bundle expansion from `summaries/variant_env_expansion.json`:

- `MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE` is active through the
  victory bundle.
- `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST` is not in the victory bundle.
- `dsv4_sm80_a100_victory_sharedbf16` expands to the same active toggles as the
  promoted path, except the shared cache env is explicit.
- `dsv4_sm80_a100_victory_metadatadeforest` remains opt-in only.

07.66 memory ledger:

| Cache | Bytes/rank | GiB/rank | KV pages/rank |
| --- | ---: | ---: | ---: |
| new shared expert BF16 cache | `270,532,608` | `0.251953` | `14.01` |
| total BF16 projection cache after 07.66 | `1,704,984,576` | `1.587891` | see 07.66 ledger |

Fresh macro memory:

| Workload | Peak allocated bytes/rank | KV cache bytes/rank max |
| --- | ---: | ---: |
| 4096/128/batch4 | `47,565,656,064` | `2,491,495,680` |
| 4096/1024/batch4 | `47,565,686,784` | `2,491,495,680` |

## Commands

Variant expansion:

```bash
python performance_milestones/target07_post_shared_expert_reprofile/scripts/collect_variant_env.py \
  --json-out performance_milestones/target07_post_shared_expert_reprofile/summaries/variant_env_expansion.json
```

4096/128 macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_shared_expert_reprofile/raw/macro_4096x128_bs4_np128 \
  --keep-going
```

4096/1024 macro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir performance_milestones/target07_post_shared_expert_reprofile/raw/macro_4096x1024_bs4_np128 \
  --keep-going
```

Rank0 nsys profile:

```bash
performance_milestones/target07_post_shared_expert_reprofile/scripts/nsys_post_shared_expert_4096x128_bs4.sh
```

Profile script settings:

- `--variants dsv4_sm80_a100_victory`
- `--prompt-len 4096 --decode-len 128 --batch-size 4`
- `--page-size 256 --num-pages 128`
- `--repeats 1 --warmup-repeats 0`
- `MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX=1`
- `MINISGL_DSV4_GRAPH_CAPTURE_NVTX=1`
- rank0 nsys capture, TP8

## Macro Performance

| Workload | 07.66 audit output tok/s | 07.67 promoted output tok/s | Delta | 07.67 decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.2034` | `62.1364` | `-0.11%` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.7707` | `131.6263` | `-0.11%` | `169.3197` | `4092` | `0` |

The promoted path matches the 07.66 audit variant within normal run-to-run
noise, so the bundle expansion gate passed before profiling.

## Nsight Profile

Artifacts:

- `raw/nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0.nsys-rep`
- `raw/nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0.sqlite`
- `summaries/nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0_direct_copy_owner.json`
- `summaries/nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0_bucket_summary.json`

Nsight workload summary:

| Metric | Value |
| --- | ---: |
| output tok/s under nsys | `50.5797` |
| decode tok/s under nsys | `151.7844` |
| graph replay | `127` |
| greedy sample replay | `127` |
| eager decode | `0` |
| decode envelope wall | `3.591306s` |
| decode envelope kernel sum | `2.959936s` |

## Direct-Copy Owner Comparison

| Metric | 07.65 promoted | 07.66 audit shared cache | 07.67 promoted |
| --- | ---: | ---: | ---: |
| total direct_copy | `0.737039s` | `0.449052s` | `0.449200s` |
| named owner coverage | `99.97%` | `99.94%` | `99.94%` |
| MoE/shared staging group | `0.379204s` | `0.097361s` | `0.096745s` |
| runner finalization trio | `0.054714s` | `0.057774s` | `0.057450s` |
| `shared_experts.gate_up_proj` | `0.165751s` | `0.000000s` | `0.000000s` |
| `shared_experts.down_proj` | `0.119724s` | `0.000000s` | `0.000000s` |

Fresh top direct-copy owners:

| Owner | 07.67 s | 07.66 s | Decision |
| --- | ---: | ---: | --- |
| `dsv4.layer*.mlp.runner.experts` | `0.054128` | `0.054546` | stable |
| `dsv4.lm_head` | `0.044396` | `0.044360` | stable |
| `dsv4.layer*.hc_ffn_pre` | `0.042060` | `0.041932` | stable |
| `dsv4.layer*.hc_attn_pre` | `0.038678` | `0.038834` | stable |
| `dsv4.layer*.mlp.runner.shared` | `0.031477` | `0.031721` | stable |
| `dsv4.layer*.attn.kv_quant` | `0.029523` | `0.029445` | stable |
| `runner_finalize_to_fp32` | `0.023019` | `0.023179` | stable |
| `runner_shared_to_fp32` | `0.022545` | `0.022688` | stable |

Conclusion: shared expert projection owners are still gone; remaining
MoE/shared finalization is still about 07.66 size; no new direct-copy owner
grew unexpectedly after promotion.

## Bucket-Level Kernel Summary

Fresh 4096/128/batch4 rank0 decode envelope:

| Bucket | Kernel s | Share of decode kernels | Count | Phase |
| --- | ---: | ---: | ---: | --- |
| projection/GEMM | `0.778887` | `26.31%` | `100,965` | mixed |
| direct-copy/layout | `0.557626` | `18.84%` | `192,641` | mixed |
| HC/elementwise | `0.536306` | `18.12%` | `212,719` | mixed |
| NCCL communication | `0.338786` | `11.45%` | `11,176` | mixed |
| MoE routed/backend | `0.300138` | `10.14%` | `43,688` | mixed |
| index/cache/topk | `0.132758` | `4.49%` | `21,646` | mixed |
| sparse attention | `0.118089` | `3.99%` | `21,590` | prefill-heavy overall |
| FP8 activation quant | `0.076019` | `2.57%` | `35,433` | mixed |
| RMSNorm/rope/compress/store | `0.072145` | `2.44%` | `24,638` | mixed |

Repeat-level view:

| Bucket | Repeat kernel s | Note |
| --- | ---: | --- |
| sparse attention | `2.226233` | dominated by prefill sparse kernel (`2.108144s` prefill, `0.118089s` decode) |
| HC/elementwise | `1.211817` | clean implementation candidate; decode is `0.536306s` |
| projection/GEMM | `1.148039` | largest decode bucket, but not selected in this target's allowed next-target set |
| direct-copy/layout | `0.964392` | broad and owner-diffuse after 07.66 |
| NCCL communication | `0.645707` | not top-two in decode; important but not first |
| MoE routed/backend | `0.562739` | backend compute, not finalization |

Top decode kernels include `ampere_sgemm_32x32_sliced1x4_tn`,
PyTorch `direct_copy_kernel_cuda`, NCCL f32/bf16 all-reduce, CUTLASS BF16 GEMM,
Marlin WNA16 MoE, `_fp8_activation_quantize_kernel`, `_hc_split_pre_kernel`,
RMSNorm reductions, sparse split-K kernels, and `gatherTopK`.

## Communication Summary

Perf matrix communication counters are identical for the two fresh macro
workloads because both run three repeats with the same prefill/decode schedule
shape:

| Label | Op | DType | Count | Bytes |
| --- | --- | --- | ---: | ---: |
| `dsv4.v1_moe_reduce_once_all_reduce` | all-reduce | fp32 | `1,376` | `369,367,187,456` |
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | all-reduce | bf16 | `1,376` | `184,683,593,728` |
| `dsv4.embedding_all_reduce` | all-reduce | bf16 | `32` | `4,294,967,296` |
| `dsv4.lm_head_all_gather` | all-gather | fp32 | `32` | `66,191,360` |
| total | mixed | mixed | `2,816` | `558,411,939,840` |

Nsight decode-envelope NCCL kernels:

| Kernel | Count | Kernel s | Mapping inference |
| --- | ---: | ---: | --- |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | `5,461` | `0.168133` | MoE reduce-once all-reduce |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | `5,588` | `0.165361` | `wo_b` all-reduce plus small embedding contribution |
| `ncclDevKernel_AllGather_RING_LL` | `127` | `0.005292` | `lm_head` all-gather |

Communication is material (`0.338786s` decode-envelope), but it is not top-two
after the fresh reprofile and the macro counters do not show a changed contract
after 07.66.

## vLLM Source Comparison

Only fresh top candidates were compared.

HC/elementwise:

- Mini uses `dsv4_kernel.hc_pre_fallback` and `hc_post_fallback` in
  `python/minisgl/models/deepseek_v4.py:1817` and `python/minisgl/models/deepseek_v4.py:1835`,
  with per-layer graph NVTX around `hc_attn_pre`, `hc_attn_post`,
  `hc_ffn_pre`, and `hc_ffn_post`.
- vLLM wraps this boundary as `torch.ops.vllm.mhc_pre` and
  `torch.ops.vllm.mhc_post` in
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:1065`
  and `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py:1090`.
- vLLM also lifts q/kv RMSNorm and `wq_b` around the attention custom-op
  boundary so Inductor can fuse adjacent residual/norm tails:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:631`
  and `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:728`.

Communication:

- Mini labels embedding, `wo_b`, MoE reduce-once, and `lm_head` collectives at
  `python/minisgl/models/deepseek_v4.py:417`,
  `python/minisgl/models/deepseek_v4.py:423`,
  and `python/minisgl/models/deepseek_v4.py:1789`.
- vLLM's fused-MoE runner has mutually exclusive early/late reduce paths in
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:338`
  and `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:357`.
- That is a real parity surface, but fresh mini evidence puts NCCL behind
  projection/GEMM, direct-copy/layout, and HC/elementwise.

Sparse attention / cache:

- vLLM has fused compress -> RMSNorm -> RoPE -> FP8 quant -> store kernels in
  `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:686`.
- It also registers custom-op wrappers for graph splitting on SM80 reference
  paths at
  `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:1021`.
- Fresh mini evidence says sparse attention is repeat-top because of prefill,
  not decode: `2.108144s` prefill vs `0.118089s` decode.

MoE runner finalization:

- vLLM has a richer fused-MoE final output/reduce contract in
  `moe_runner.py:580`.
- Mini's remaining finalization trio is only `0.057450s`, below the target's
  `0.15s` implementation gate.

## Next Target Recommendation

Recommend exactly one next implementation target:

```text
HC / elementwise graph cleanup
```

Evidence:

- Fresh decode-envelope HC/elementwise bucket is `0.536306s`.
- Fresh repeat HC/elementwise bucket is `1.211817s`.
- Adjacent direct-copy/layout is also large (`0.557626s` decode,
  `0.964392s` repeat), but direct-copy owner attribution is diffuse and stable;
  the clean source-aligned candidate is the HC/elementwise graph boundary.
- vLLM has an explicit `mhc_pre` / `mhc_post` custom-op boundary and compile
  strategy for the same region.
- Runner finalization is too small (`0.057450s`) and communication is not
  top-two (`0.338786s`) in the fresh decode profile.

Expected upside:

- Target a `0.15s-0.25s` reduction in the 4096/128 rank0 decode-envelope
  HC/elementwise plus adjacent layout bucket.
- Expected 4096/1024 macro upside: `>=3%` only if the decode-envelope reduction
  survives without increasing projection/GEMM, NCCL, or MoE backend time.

Stop gates:

- Keep it opt-in; do not add it to `dsv4_sm80_a100_victory` until macro and
  profile gates pass.
- Exactness gate: numerical parity against current promoted path for HC pre/post
  and surrounding RMSNorm/residual boundaries.
- Runtime gate: graph replay stays active and eager decode stays `0`.
- Profile gate: reduce 4096/128 rank0 HC/elementwise plus adjacent layout by
  `>=0.15s`, or stop.
- Macro gate: 4096/1024/batch4 output tok/s improves by `>=3%`, or stop.
- Scope gate: no precision-route change, no NCCL contract change, no sparse
  attention rewrite, no MoE runner finalization rewrite in this next target.

## Why Not Other Candidates

Communication parity / reduce-contract:

- NCCL is `0.338786s` in the fresh decode envelope, not top-two.
- Perf matrix communication counters are stable and expected:
  MoE fp32 reduce-once and `wo_b` bf16 all-reduce dominate.
- Defer until NCCL becomes top-two or a contract difference is isolated.

MoE runner finalization cleanup:

- Finalization trio is `0.057450s`, below the `0.15s` implementation gate.
- The larger MoE bucket is routed/backend compute (`0.300138s` decode), not the
  fp32 finalization boundary.

Shared expert overlap audit:

- Shared projection direct-copy owners remain eliminated.
- Remaining `runner.shared` direct-copy is stable at `0.031477s`.
- Shared overlap may be useful later, but fresh evidence does not make it the
  next bottleneck.

Sparse attention / prefill-decode split:

- Repeat sparse attention is large (`2.226233s`) but prefill-heavy.
- Decode sparse attention is only `0.118089s`, below the implementation gate.
- For 4096/1024 decode throughput, the prefill cost is amortized; do not start
  an attention rewrite from this profile.

INT8 MoE feasibility:

- This target did not choose a precision route.
- The fresh exact-path evidence points to HC/elementwise cleanup first.
- INT8 would need separate quality gates and should not be mixed with an exact
  graph cleanup target.

Projection/GEMM:

- Projection/GEMM is the largest fresh decode bucket (`0.778887s`), but it is
  not one of the approved next categories for this reset.
- Previous targets already handled exact BF16 projection-cache staging.  Further
  projection work likely becomes backend/precision work, so it should be
  revisited only after an explicit target is defined.

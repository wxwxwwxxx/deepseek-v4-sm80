# TARGET 07.69: Projection/GEMM Backend Owner Re-Attribution

Date: 2026-07-02

## Scope

This target is measurement and decision only.  It did not implement a new
projection/GEMM kernel, did not promote any opt-in path, and did not change the
default precision route.

Baseline:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- model: `/models/DeepSeek-V4-Flash`
- page size: `256`
- TP8, 8x A100
- source profile: TARGET 07.67 promoted rank0 4096/128/batch4 Nsight SQLite

Repo state at report time:

```text
## dsv4-sglang-based...origin/main [ahead 42]
?? performance_milestones/target07_projection_gemm_backend_owner_reattribution/
```

No runtime source files were changed.  New files are milestone-local scripts,
summaries, symlinks, and this README.

## Current Macro

Promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

Active promoted stack includes the 07.66 shared-expert BF16 cache:

```text
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1
```

The TARGET 07.64 metadata deforestation and TARGET 07.68 HC graph cleanup
paths are still opt-in only and are not used as this baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1  # not promoted
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1          # not promoted
```

07.68 same-run profile confirmed projection/GEMM was unchanged by HC cleanup:
`0.778887s -> 0.779055s`, so 07.67 promoted profile remains valid for
projection/GEMM owner attribution.

## Evidence Reuse

Existing 07.67/07.68 profiles were sufficient.  No
`MINISGL_DSV4_PROFILE_PROJECTION_NVTX` flag was added and no fresh Nsight
capture was needed.

The classifier maps replayed CUDA graph kernels through `originalGraphNodeId`
back to capture-time `dsv4.*` NVTX ranges, then groups GEMM-like kernel names
by owner and backend family.

Artifacts:

- `raw/baseline_0767_rank0.sqlite` -> symlink to the 07.67 promoted profile
- `raw/baseline_0767_rank0.nsys-rep` -> symlink to the 07.67 promoted profile
- `raw/optin_0768_hccleanup_rank0.sqlite` -> symlink to the 07.68 opt-in profile
- `summaries/projection_gemm_owner_table.md`
- `summaries/projection_gemm_backend_families.md`
- `summaries/projection_gemm_top_kernels.md`
- `summaries/vllm_projection_source_parity.md`
- `raw/focused_projection_microbench.md`

Commands:

```bash
python -m py_compile \
  performance_milestones/target07_projection_gemm_backend_owner_reattribution/scripts/classify_projection_gemm_owners.py \
  performance_milestones/target07_projection_gemm_backend_owner_reattribution/scripts/focused_projection_microbench.py

python performance_milestones/target07_projection_gemm_backend_owner_reattribution/scripts/classify_projection_gemm_owners.py \
  --sqlite performance_milestones/target07_post_shared_expert_reprofile/raw/nsys_target0767_dsv4_sm80_a100_victory_4096x128_bs4_np128_rank0.sqlite \
  --output-dir performance_milestones/target07_projection_gemm_backend_owner_reattribution/summaries \
  --top 50

CUDA_VISIBLE_DEVICES=0 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
python performance_milestones/target07_projection_gemm_backend_owner_reattribution/scripts/focused_projection_microbench.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 9 \
  --tokens 1 4 8 16 \
  --warmup 50 \
  --iters 200 \
  --output performance_milestones/target07_projection_gemm_backend_owner_reattribution/raw/focused_projection_microbench.json
```

## Attribution Gate

4096/128/batch4 rank0 decode envelope:

| Metric | Value |
| --- | ---: |
| Decode envelope wall | `3.591306s` |
| Projection/GEMM bucket | `0.778887s` |
| Named/grouped owner time | `0.770601s` |
| Residual/coarse owner time | `0.008286s` |
| Named coverage | `98.94%` |

This passes the stop rule: it explains more than `80%` of the projection/GEMM
bucket and more than `0.60s` of owner/backend time.

## Owner Table

| Owner group | Kernel s | Share | Backend family | Decision |
| --- | ---: | ---: | --- | --- |
| HC pre linear | `0.178373` | `22.90%` | cuBLAS SGEMM/FP32 + splitK/reduce | Largest grouped owner, but below single-owner `0.20s` gate. |
| attention WQA/WKV/compress | `0.119458` | `15.34%` | CUTLASS BF16 + splitK/reduce | Top BF16-cluster owner group. |
| MoE router / route projection | `0.097109` | `12.47%` | cuBLAS SGEMM/FP32 + splitK/reduce | Track; below owner gate. |
| shared experts cached BF16 | `0.085848` | `11.02%` | CUTLASS BF16 + splitK/reduce | 07.66 removed staging; compute remains diffuse. |
| attention `wo_a` | `0.063857` | `8.20%` | cuBLASLt BF16 + splitK/reduce | Cached BF16 BMM residual. |
| attention `q_wqb` | `0.056392` | `7.24%` | cuBLASLt BF16 GEMM | Cached BF16 residual. |
| attention `wo_b` local | `0.054507` | `7.00%` | cuBLASLt BF16 GEMM | Local projection only; communication is separate. |
| indexer weight/compressor projection | `0.043647` | `5.60%` | CUTLASS/cuBLASLt BF16 + splitK/reduce | Track inside BF16 cluster. |
| indexer `wq_b` | `0.042727` | `5.49%` | cuBLASLt BF16 GEMM | Cached BF16 residual. |
| `lm_head` | `0.026769` | `3.44%` | cuBLAS SGEMM/FP32 | Too small. |
| residual / coarse owner | `0.008286` | `1.06%` | mixed | Small enough; no recapture needed. |
| model HC head/expand | `0.001915` | `0.25%` | cuBLAS SGEMM/FP32 + splitK/reduce | Too small. |

Raw graph-owner details are in
`summaries/projection_gemm_owner_table.md`.

Important read: no single owner clears `0.20s`.  The old 07.57
`_quantized_linear_fp8_kernel` owner table is stale; the promoted path now has
zero meaningful residual `_quantized_linear_fp8_kernel` time in this bucket.

## Backend Families

| Backend cluster/family | Kernel s | Share | Main owners | Gate read |
| --- | ---: | ---: | --- | --- |
| BF16 small-GEMM + splitK/reduce cluster | `0.521619` | `66.97%` | attention WQA/WKV/compress, shared experts, `wo_a`, `q_wqb`, `wo_b`, indexer | Clears same-backend cluster gate. |
| FP32/SGEMM small-GEMM cluster | `0.257269` | `33.03%` | HC pre linear, MoE router, `lm_head` | Below same-backend cluster gate. |
| cuBLAS SGEMM/FP32 GEMM | `0.257269` | `33.03%` | HC pre, MoE router, `lm_head` | Track but not enough alone. |
| cuBLASLt BF16 GEMM | `0.219912` | `28.23%` | `q_wqb`, `wo_b`, `wo_a`, indexer `wq_b` | Part of BF16 cluster. |
| CUTLASS BF16 GEMM | `0.194319` | `24.95%` | WQA/WKV/compress, shared experts | Part of BF16 cluster. |
| cuBLASLt splitK/reduce | `0.107388` | `13.79%` | HC pre, WQA/WKV, shared experts, indexer, route | Attribute to parent GEMMs; do not optimize reduce alone first. |
| residual FP8 quantized linear | `0.000000` | `0.00%` | none observed in promoted bucket | Old 07.57 target is not current. |

Full backend table:
`summaries/projection_gemm_backend_families.md`.

## Focused Microbench

The microbench used real layer-9 DSV4 weights and measured only
profile-selected owners/backend representatives: HC pre linear, MoE router,
and the top BF16-cluster attention WQA/WKV owner.  It did not benchmark all
projection modules.

Device: one A100-SXM4-80GB, CUDA capability `(8, 0)`, PyTorch `2.9.1+cu128`,
`torch.backends.cuda.matmul.allow_tf32=False`.

| Case | Owner | M=1 mean ms | M=4 mean ms | M=8 mean ms | M=16 mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `hc_attn_pre_linear_bf16_fp32_fallback` | HC pre linear | `0.056493` | `0.058148` | `0.057837` | `0.058296` |
| `hc_ffn_pre_linear_bf16_fp32_fallback` | HC pre linear | `0.056094` | `0.057883` | `0.057820` | `0.058168` |
| `moe_router_gate_linear` | MoE router / route projection | `0.062460` | `0.067730` | `0.068906` | `0.068772` |
| `attn_qproj_fused_wqa_wkv_cached_bf16_gemm_only` | attention WQA/WKV/compress | `0.036729` | `0.042605` | `0.042849` | `0.043011` |
| `attn_qproj_fused_wqa_wkv_cached_bf16_with_act_quant` | attention WQA/WKV/compress | `0.102567` | `0.109819` | `0.109257` | `0.109605` |

Readout:

- HC pre and MoE router FP32 SGEMMs are launch/dispatch dominated and almost
  flat across `M=1..16`, but together they still do not clear the backend
  cluster gate.
- The q_proj GEMM-only BF16 representative is also flat across decode-small
  `M`; adding the promoted FP8 activation quant helper raises the local
  boundary from about `0.043 ms` to about `0.110 ms` at `M=4`.
- Because the profile bucket counted activation quant separately
  (`0.076019s` in 07.67), the next projection/GEMM target should keep its
  primary denominator to BF16 GEMM kernels and treat activation quant only as
  boundary context.

Full microbench artifact:
`raw/focused_projection_microbench.md`.

## vLLM Source Parity

Source parity is summarized in
`summaries/vllm_projection_source_parity.md`.

Key conclusions:

- vLLM is not simply running the same exact BF16 projection contract faster.
  Its DeepSeek V4 path uses `deepseek_v4_fp8`, quantized linear layers, packed
  FP8 KV/indexer cache machinery, `torch.ops.vllm.deepseek_v4_attention`,
  `fused_inv_rope_fp8_quant`, `deepseek_v4_fp8_einsum`, and
  `@support_torch_compile` boundaries.
- HC pre is the closest source-level contract to mini's exact route: both have
  FP32-like matmul on SM80, but vLLM's `mhc_pre` returns FP32 `post/comb` while
  mini promoted path keeps BF16 `post/comb`.  07.68 already tested a mini
  opt-in HC cleanup and did not promote it.
- vLLM `wo_a` has an SM80 BF16 BMM reference path similar to mini's promoted
  cached BF16 BMM, but the fast path can route through FP8 inverse-RoPE/einsum
  before `wo_b`.
- vLLM WQA/WKV, `q_wqb`, indexer, shared experts, and MoE ownership boundaries
  generally use quantized/custom-op/runner contracts, not a drop-in exact BF16
  cache replacement.

Therefore vLLM source supports a backend-boundary hypothesis, but not an
unqualified precision-neutral port.

## Decision

Decision:

- Next target: `TARGET 07.70 DSV4 SM80 BF16 Small-GEMM Backend Cluster`
- Primary owner/backend: BF16 small-GEMM + cuBLASLt splitK/reduce cluster
  (`0.521619s`, `66.97%` of current projection/GEMM)
- Initial focused owners: attention WQA/WKV/compress, `wo_a`, `q_wqb`, `wo_b`,
  shared experts, and indexer cached BF16 projections; keep HC pre/router as
  context, not the primary implementation surface.
- Expected profile gain: require at least `0.10s` rank0 decode-envelope
  projection/GEMM reduction before any promotion discussion; a stronger target
  should aim for `0.12-0.16s` by reducing BF16 small-GEMM launch/backend
  overhead across the cluster.
- Expected 4096/1024 macro gain: only credible if the BF16 cluster drops by
  roughly `20-30%`; target gate should require at least `+3%` same-run
  4096/1024 output tok/s with graph replay preserved and eager decode `0`.
- Why not precision yet: the current promoted route is exact BF16, and vLLM's
  faster projection-adjacent mechanisms are entangled with FP8/MXFP4/cache and
  compile boundaries.  The BF16 backend cluster is still large enough to test
  an exact backend target first.
- Why not single-owner kernel now: no owner group reaches `0.20s`; the largest
  is HC pre linear at `0.178373s`, while the selected same-backend cluster
  clears the `0.35s` gate.

Stop condition for the next target:

- Stop without a backend implementation if focused real-shape BF16 owner
  microbenchmarks cannot show at least `15%` latency reduction on profile
  shapes for two or more cluster representatives.
- Stop without promotion if the fresh 4096/128 rank0 profile does not reduce
  projection/GEMM by at least `0.10s`, or if 4096/1024 same-run macro does not
  improve by at least `3%`.
- Stop and write a precision-policy target instead if the only plausible vLLM
  parity mechanism requires FP8/MXFP4 carrier, FP32 `post/comb`, packed cache
  layout, or broader compile/runtime ownership changes.

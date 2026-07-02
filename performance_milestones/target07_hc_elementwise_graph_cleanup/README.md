# TARGET 07.68: DSV4 SM80 HC / Elementwise Graph Boundary Cleanup

Date: 2026-07-02

## Scope

This target stayed on the promoted exact BF16 route:

- variant: `dsv4_sm80_a100_victory`
- bundle: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- opt-in candidate: `MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1`
- candidate variant: `dsv4_sm80_a100_victory_hccleanup`

No projection/GEMM backend, NCCL/communication, sparse attention, MoE runner
finalization, shared expert overlap, INT8/other precision route, or TARGET
07.64 metadata deforestation promotion was done.

Current git status is recorded in
`summaries/git_status_current.txt`.  The inherited 07.67 bucket and
direct-copy owner summaries are copied to:

- `summaries/baseline_0767_bucket_summary.md`
- `summaries/baseline_0767_direct_copy_owner.md`

## Source Parity

| Boundary | mini current | vLLM reference | Adapt decision |
| --- | --- | --- | --- |
| HC pre RMS/rsqrt | `hc_pre_fallback` does `flat.float().square().mean().rsqrt()` as PyTorch graph nodes before HC split. | `_mhc_pre_fused_triton` computes squared sum and rsqrt inside the fused pre kernel on SM80 reference path. | Ported this boundary into a mini-owned opt-in Triton kernel. |
| HC pre matmul | `linear_bf16_fp32_fallback(flat, fn)` is preserved; promoted path uses BF16 cached HC weights but keeps the existing linear contract. | SM80 reference uses one matmul before `_mhc_pre_fused_triton`; non-reference can use deeper fused GEMM/Triton/TileLang paths. | Left unchanged to avoid projection/GEMM/backend or precision route work. |
| HC split/Sinkhorn/pre output | `_hc_split_pre_kernel` produces `y`, `post`, `comb`; `post/comb` are BF16 and the kernel is launched per `(token, hidden block)`. | vLLM computes `pre/post/comb` once per token, then a layer-input kernel; `post/comb` are FP32. | Implemented a two-kernel mini variant, but kept mini's BF16 `post/comb` contract. |
| HC post | mini already has `_hc_post_kernel` under `MINISGL_DSV4_SM80_HC`. | vLLM has `_mhc_post_triton` consuming FP32 `post/comb`. | Left unchanged; direct vLLM dtype contract would change promoted BF16 path. |
| post/comb dtype | BF16 in mini promoted path. | FP32 in vLLM reference. | Kept BF16 for exact-path compatibility. |
| layout/materialization | Current path includes RMS PyTorch launches and HC-pre direct-copy/layout work around `mixes.contiguous()` and dtype staging. | vLLM custom op boundary removes most eager RMS/Sinkhorn graph nodes. | Candidate removes RMS PyTorch nodes and reduces HC-pre direct-copy owner time, but does not change linear casts/backend. |

## Implementation

Added:

- `DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE` in `python/minisgl/kernel/deepseek_v4.py`.
- An opt-in branch in `hc_pre_fallback` that calls `hc_prenorm_split_pre`.
- Triton helpers in `python/minisgl/kernel/triton/deepseek_v4.py`:
  - `_hc_prenorm_split_pre_kernel`
  - `_hc_layer_input_kernel`
  - `hc_prenorm_split_pre`
- benchmark/text-smoke variant `dsv4_sm80_a100_victory_hccleanup`.
- focused HC microbench script:
  `scripts/focused_hc_microbench.py`
- nsys profile script:
  `scripts/nsys_hc_cleanup_4096x128_bs4.sh`

The new toggle is in `DSV4_SM80_EXPERIMENTAL_TOGGLES` and is deliberately not
in `DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST`.

## Commands

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  performance_milestones/target07_hc_elementwise_graph_cleanup/scripts/focused_hc_microbench.py

pytest -q -o addopts='' \
  tests/benchmark/test_deepseek_v4_perf_matrix.py::test_configure_variant_records_hc_graph_cleanup \
  tests/benchmark/test_deepseek_v4_text_smoke.py::test_configure_variant_sets_hc_graph_cleanup

pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy \
  tests/kernel/test_deepseek_v4_wrappers.py::test_hc_graph_cleanup_opt_in_matches_current_hc_path

python performance_milestones/target07_hc_elementwise_graph_cleanup/scripts/focused_hc_microbench.py \
  --output performance_milestones/target07_hc_elementwise_graph_cleanup/raw/focused_hc_microbench.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
  --output performance_milestones/target07_hc_elementwise_graph_cleanup/raw/text_smoke.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 128 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_hc_elementwise_graph_cleanup/raw/macro_4096x128_bs4_np128 \
  --keep-going

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 --decode-len 1024 --batch-size 4 \
  --repeats 3 --warmup-repeats 1 \
  --page-size 256 --num-pages 128 \
  --output-dir performance_milestones/target07_hc_elementwise_graph_cleanup/raw/macro_4096x1024_bs4_np128 \
  --keep-going

performance_milestones/target07_hc_elementwise_graph_cleanup/scripts/nsys_hc_cleanup_4096x128_bs4.sh
```

## Focused HC Microbench

Shape: `tokens=4`, `hc_mult=4`, `hidden=4096`, `mix_hc=24`,
`sinkhorn_iters=20`, BF16 `x/fn`, FP32 `scale/base`.

| Path | hc_pre ms | hc_pre CUDA kernels | hc_post ms | hc_post CUDA kernels | Output dtype/shape | Error vs current |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| current mini | `0.247450` | `11` | `0.046508` | `1` | `y=[4,4096]`, `post=[4,4]`, `comb=[4,4,4]`, BF16 | reference |
| hccleanup | `0.196063` | `6` | `0.052608` | `1` | same | max/mean/p99 abs all `0.0` for `y/post/comb/post_out` |

Focused result: HC pre improved by about `20.77%` and removed five CUDA kernel
events per call in the microbench.

## Correctness

- Unit/config tests passed.
- Focused HC output comparison passed against the current HC path.
- TP8 text smoke passed for both variants.
- Text smoke hccleanup graph replay: `18`, eager decode: `0`.
- Macro graph replay remained active:
  - 4096/128: `1016`, eager `0`
  - 4096/1024: `8184`, eager `0`

Text smoke outputs matched the baseline prompts:

- `2 + 2 等于 4。`
- `The sky is blue on a clear day.`
- `杭州是风景如画的历史文化名城。`

## Macro

07.67 promoted baseline:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

This target, same-run comparison:

| Workload | Variant | Output tok/s | Decode tok/s | Delta vs same-run victory | Graph replay | Eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | victory | `62.2789` | `169.0882` | reference | `1016` | `0` |
| 4096/128/batch4 | hccleanup | `64.3335` | `169.2758` | `+3.30%` | `1016` | `0` |
| 4096/1024/batch4 | victory | `131.5675` | `168.8894` | reference | `8184` | `0` |
| 4096/1024/batch4 | hccleanup | `132.5223` | `168.6270` | `+0.73%` | `8184` | `0` |

Long macro vs inherited 07.67 output tok/s is `+0.68%`, below the `+3%`
promotion gate.

## Nsys Bucket Comparison

4096/128/batch4 rank0 decode envelope:

| Bucket | 07.67 promoted s | hccleanup s | Delta |
| --- | ---: | ---: | ---: |
| HC/elementwise | `0.536306` | `0.519303` | `-0.017003` (`-3.17%`) |
| direct-copy/layout | `0.557626` | `0.520151` | `-0.037475` |
| projection/GEMM | `0.778887` | `0.779055` | `+0.000168` |
| NCCL communication | `0.338786` | `0.338458` | `-0.000328` |
| MoE routed/backend | `0.300138` | `0.299245` | `-0.000893` |

Representative hccleanup decode kernels:

- `_hc_prenorm_split_pre_kernel`: `0.144871s`
- `_hc_layer_input_kernel`: `0.038542s`
- `_hc_post_kernel`: `0.036981s`

The PyTorch `pow/mean/rsqrt` HC pre chain is removed from the top decode kernel
list, but the new prenorm kernel cost means the full HC bucket reduction is
small.

## Direct-Copy Owner Comparison

4096/128/batch4 rank0 direct-copy owners:

| Owner | 07.67 s | hccleanup s | Delta |
| --- | ---: | ---: | ---: |
| `dsv4.layer*.hc_ffn_pre` | `0.042060` | `0.023566` | `-0.018494` |
| `dsv4.layer*.hc_attn_pre` | `0.038678` | `0.020875` | `-0.017803` |
| `dsv4.model.hc_head` | `0.001110` | `0.001120` | `+0.000010` |
| `dsv4.model.hc_expand` | `0.000403` | `0.000402` | `-0.000001` |

HC pre owner direct-copy combined:

- 07.67: `0.080738s`
- hccleanup: `0.044441s`
- delta: `-0.036297s`

Combined decode HC/elementwise plus HC-owned pre direct-copy reduction:
`0.017003s + 0.036297s = 0.053300s`, below the `0.15s` profile gate.

## Decision

Decision: keep opt-in, do not promote.

Reasons:

- Correctness passed.
- Focused microbench and short macro show useful local signal.
- Profile gate failed:
  - HC/elementwise bucket reduced only `3.17%`, below `25%`.
  - HC/elementwise + HC-owned layout reduced about `0.0533s`, below `0.15s`.
- Macro promotion gate failed:
  - 4096/1024 output tok/s improved only `+0.73%` vs same-run victory and
    `+0.68%` vs 07.67, below required `+3%`.
- 4096/128 did not regress; it improved `+3.30%`.

The opt-in path remains useful for future HC experiments because it establishes
a tested mini-owned MHC boundary while preserving BF16 `post/comb`.

## Next Steps

- Do not spend more time on scattered generic elementwise cleanup in this
  target.
- A stronger HC target would need to reduce the prenorm kernel cost, likely by
  moving squared-sum/prenorm closer to the matmul backend while keeping exact
  BF16 carrier semantics.
- Direct vLLM port is blocked for promotion by the `post/comb` dtype mismatch
  unless a separate exactness decision explicitly accepts FP32 carrier.
- The remaining top decode bucket is still projection/GEMM. A future target
  should treat that as backend/precision work rather than HC cleanup.

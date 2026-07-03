# TARGET 07.68: DSV4 SM80 HC / Elementwise Graph Boundary Cleanup

Date: 2026-07-02

## Goal

Reduce the current promoted exact-path HC / elementwise graph bucket after
TARGET 07.67.

This is an exact bf16-path implementation target.  It should focus on the
hidden-carrier (`HC`) pre/post and adjacent elementwise/layout boundaries in
DeepSeek V4 decoder layers.  Do not change precision routes, communication
contracts, sparse attention, MoE runner finalization, or projection/GEMM
backends in this target.

The main question is whether mini can adapt or reproduce the vLLM-style
`mhc_pre` / `mhc_post` boundary to remove launches and layout/materialization
around HC without changing model numerics.

## Current Promoted Baseline

Use the promoted variant:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Current confirmed promoted macro from TARGET 07.67:

| Workload | Output tok/s | Decode tok/s | Graph replay | Eager decode |
| --- | ---: | ---: | ---: | ---: |
| 4096/128/batch4 | `62.1364` | `168.6702` | `508` | `0` |
| 4096/1024/batch4 | `131.6263` | `169.3197` | `4092` | `0` |

TARGET 07.66 is already promoted:

```text
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE=1
```

The 07.64 metadata deforestation path remains opt-in only and must not be
added to the victory bundle in this target.

## Starting Evidence From TARGET 07.67

Fresh 4096/128/batch4 rank0 decode envelope:

| Bucket | Kernel s | Share | Decision |
| --- | ---: | ---: | --- |
| projection/GEMM | `0.778887` | `26.31%` | largest bucket, but likely backend/precision work |
| direct-copy/layout | `0.557626` | `18.84%` | broad and owner-diffuse |
| HC/elementwise | `0.536306` | `18.12%` | selected exact cleanup target |
| NCCL communication | `0.338786` | `11.45%` | important, but not top-two |
| MoE routed/backend | `0.300138` | `10.14%` | backend compute, not finalization |

Repeat-level view:

| Bucket | Repeat kernel s | Note |
| --- | ---: | --- |
| sparse attention | `2.226233` | prefill-heavy: `2.108144s` prefill vs `0.118089s` decode |
| HC/elementwise | `1.211817` | selected exact-path candidate |
| projection/GEMM | `1.148039` | still important, but not this target |
| direct-copy/layout | `0.964392` | adjacent to HC but diffuse |
| NCCL communication | `0.645707` | defer unless it becomes top-two |

HC-related direct-copy owners from TARGET 07.67:

| Owner | Direct-copy s | Decision |
| --- | ---: | --- |
| `dsv4.layer*.hc_ffn_pre` | `0.042060` | source-aligned HC boundary |
| `dsv4.layer*.hc_attn_pre` | `0.038678` | source-aligned HC boundary |
| `dsv4.model.hc_head` | `0.001110` | too small by itself |
| `dsv4.model.hc_expand` | `0.000403` | too small by itself |

Representative HC / elementwise kernels in the fresh profile:

| Kernel | Decode s | Repeat s | Note |
| --- | ---: | ---: | --- |
| `_hc_split_pre_kernel` | `0.070713` | `0.424397` | current mini fused HC split/pre kernel |
| `_hc_post_kernel` | `0.036269` | `0.105966` | current mini fused HC post kernel |
| `pow_tensor_scalar_kernel` | `0.023491` | `0.130129` | RMS/variance-style pre-HC math |
| `MeanOps` reduce kernel | `0.064230` | `0.117757` | RMS/variance-style pre-HC math |
| `rsqrt_kernel_cuda` | `0.023456` | see bucket | RMS/variance-style pre-HC math |

The selected target is not "all elementwise everywhere."  It is specifically
the HC pre/post region and adjacent layout/materialization that vLLM handles as
named custom-op boundaries.

## Important Caveat

Projection/GEMM is still the largest decode bucket after TARGET 07.67.  It is
not selected here because previous exact BF16 projection-cache targets already
removed the most obvious exact staging wins, and further projection work likely
becomes backend or precision work.

This target chooses HC/elementwise because it is a cleaner exact-path boundary
with a vLLM source analogue.  The final README must preserve this nuance.  Do
not claim projection/GEMM is solved.

## Mini Source Boundaries

Primary mini files:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Key mini code path:

- `DeepseekV4DecoderLayer._hc_pre`
  calls `dsv4_kernel.hc_pre_fallback`.
- `DeepseekV4DecoderLayer._hc_post`
  calls `dsv4_kernel.hc_post_fallback`.
- `DeepseekV4DecoderLayer.forward`
  runs `hc_attn_pre`, attention norm/attention, `hc_attn_post`,
  `hc_ffn_pre`, FFN norm/MoE, and `hc_ffn_post`.
- `hc_pre_fallback` currently performs:

  ```text
  flat = x.flatten(1)
  flat_float = flat.float()
  rsqrt = torch.rsqrt(flat_float.square().mean(-1, keepdim=True) + norm_eps)
  mixes = linear_bf16_fp32_fallback(flat, fn) * rsqrt
  hc_split_pre(mixes.contiguous(), x, scale, base, ...)
  ```

- `hc_post_fallback` already has a Triton `hc_post` path when
  `MINISGL_DSV4_SM80_HC` is active.
- `python/minisgl/kernel/triton/deepseek_v4.py` contains
  `_hc_split_pre_kernel`, `_hc_post_kernel`, `hc_split_pre`, and `hc_post`.

The current mini implementation already has some HC Triton kernels, so this
target is not starting from eager PyTorch only.  The likely remaining issue is
that pre-HC RMS/variance, matmul scaling, `mixes.contiguous()`, and split/pre
work are still separated into multiple graph nodes.

## vLLM Reference

Relevant vLLM files:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py
```

vLLM model boundary:

- `DeepseekV4DecoderLayer.hc_pre` calls `torch.ops.vllm.mhc_pre`.
- `DeepseekV4DecoderLayer.hc_post` calls `torch.ops.vllm.mhc_post`.
- The forward path uses:

  ```text
  x, post, comb = self.hc_pre(...)
  x = self.attn_norm(x)
  x = self.attn(...)
  x = self.hc_post(x, residual, post, comb)

  x, post, comb = self.hc_pre(...)
  x = self.ffn_norm(x)
  x = self.ffn(...)
  x = self.hc_post(x, residual, post, comb)
  ```

vLLM `layers/mhc.py` notes that the SM80 reference `mhc_pre` path is:

- one cuBLAS matmul;
- a fused Triton `mhc_pre` kernel;
- around two kernels instead of a long PyTorch chain;
- intended to replace many launches in the reference path.

The first task is source parity: determine whether mini can directly adapt
vLLM's `mhc_pre` / `mhc_post` kernel logic or whether mini's current kernels
are close and only need boundary/layout changes.

## Scope

In scope:

- source-level parity review of mini HC vs vLLM `mhc_pre` / `mhc_post`;
- focused microbenchmarks for `hc_pre` and `hc_post` on real DSV4 shapes;
- owner/bucket attribution for HC pre-HC RMS/variance, matmul, split/pre,
  post, and adjacent layout/copy;
- one narrow opt-in implementation that reduces HC/elementwise graph cost;
- adding benchmark/text-smoke variants for the opt-in;
- correctness comparison against the current promoted path;
- final macro and nsys reprofile.

Out of scope:

- projection/GEMM backend changes;
- NCCL / communication contract changes;
- sparse attention rewrites;
- MoE runner finalization cleanup;
- shared expert overlap;
- INT8 MoE or any new precision path;
- full FP8 KV cache or `fp8_ds_mla`;
- broad torch.compile experiments outside HC/residual/norm boundaries;
- changing the 07.64 metadata deforestation status.

## Candidate Implementation Directions

Try candidates in this order.  Stop once one candidate clears the gates.

### 1. vLLM MHC Boundary Adaptation Probe

Audit `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py` and decide
whether mini should port/adapt the fused `mhc_pre` and `mhc_post` kernels.

Questions:

- Does vLLM keep `post` and `comb` in fp32 while mini stores bf16?
- Does vLLM fuse RMS/variance into the post-matmul Triton kernel while mini
  computes it separately before `hc_split_pre`?
- Does vLLM avoid `mixes.contiguous()` or other repeated layout work?
- Are output shapes and dtypes compatible with mini's decoder layer?
- Can mini keep exactness relative to the current promoted path, or is vLLM's
  precision subtly different?

If direct port is plausible, implement it as an opt-in mini-owned path.  Do not
import vLLM as a runtime dependency.

### 2. Mini-Owned `hc_pre` Boundary Fusion

If direct vLLM port is too invasive, implement a mini-owned opt-in that fuses
or reduces the current `hc_pre_fallback` graph nodes.

Possible cuts:

- fuse `flat.float().square().mean().rsqrt()` with the HC split/pre kernel
  contract where practical;
- reduce or eliminate `mixes.contiguous()` if a kernel can consume the current
  layout safely;
- keep the BF16/FP32 dtype contract explicit and test output deltas;
- preserve the existing `linear_bf16_fp32_fallback` behavior unless the target
  explicitly proves a better exact equivalent.

### 3. HC Post/Layout Cleanup

If `hc_pre` has already reached vLLM-like shape or the profile shows `hc_post`
dominates after a first cut, test a smaller `hc_post` / output-layout cleanup.

This is secondary because 07.67 shows the broader HC/elementwise bucket is
more than the post kernel alone.

## Work Plan

1. Create artifacts:

   ```text
   performance_milestones/target07_hc_elementwise_graph_cleanup/
     README.md
     raw/
     summaries/
     scripts/
   ```

2. Freeze inherited baseline.

   Record:

   - current git state;
   - `dsv4_sm80_a100_victory` macro from 07.67;
   - 07.67 bucket summary;
   - 07.67 direct-copy owner summary;
   - active HC and shared expert cache toggles;
   - note that 07.64 metadata deforestation remains disabled.

3. Build focused HC probes.

   Add scripts or tests that measure real-shape `hc_pre` and `hc_post`
   boundaries using loaded model weights where possible.  At minimum report:

   - current mini `hc_pre` time by shape;
   - current mini `hc_post` time by shape;
   - number of CUDA kernels launched per call if measurable;
   - output dtype/shape;
   - max/mean error against current promoted path for any new candidate.

4. Compare mini and vLLM source contracts.

   Produce a table:

   | Boundary | mini current | vLLM reference | Adapt decision |
   | --- | --- | --- | --- |
   | HC pre RMS/rsqrt | ... | ... | ... |
   | HC pre matmul | ... | ... | ... |
   | HC split/sinkhorn/pre output | ... | ... | ... |
   | HC post | ... | ... | ... |
   | post/comb dtype | ... | ... | ... |
   | layout/materialization | ... | ... | ... |

5. Implement one opt-in candidate.

   Suggested toggle and variant names:

   ```text
   MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
   dsv4_sm80_a100_victory_hccleanup
   ```

   Add the variant to:

   ```text
   benchmark/offline/deepseek_v4_perf_matrix.py
   benchmark/offline/deepseek_v4_text_smoke.py
   ```

   Keep the promoted `dsv4_sm80_a100_victory` unchanged until all gates pass.

6. Validate correctness.

   Required:

   - unit tests for the new HC helper or path;
   - focused output comparison against the current HC path;
   - TP8 text smoke with page size 256;
   - graph replay remains active and eager decode remains `0`.

   Suggested text smoke:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_text_smoke.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
     --output performance_milestones/target07_hc_elementwise_graph_cleanup/raw/text_smoke.json
   ```

7. Run macro benchmarks.

   Short macro:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
     --scenarios decode_throughput_bs8 \
     --prompt-len 4096 \
     --decode-len 128 \
     --batch-size 4 \
     --repeats 3 \
     --warmup-repeats 1 \
     --page-size 256 \
     --num-pages 128 \
     --output-dir performance_milestones/target07_hc_elementwise_graph_cleanup/raw/macro_4096x128_bs4_np128 \
     --keep-going
   ```

   Long macro if short macro or owner profile improves:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   torchrun --standalone --nproc_per_node=8 \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     --model-path /models/DeepSeek-V4-Flash \
     --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_hccleanup \
     --scenarios decode_throughput_bs8 \
     --prompt-len 4096 \
     --decode-len 1024 \
     --batch-size 4 \
     --repeats 3 \
     --warmup-repeats 1 \
     --page-size 256 \
     --num-pages 128 \
     --output-dir performance_milestones/target07_hc_elementwise_graph_cleanup/raw/macro_4096x1024_bs4_np128 \
     --keep-going
   ```

8. Capture final nsys profile.

   Capture 4096/128/batch4 rank0 for the opt-in variant with graph/source NVTX
   enabled.  Compare against 07.67:

   - HC/elementwise bucket;
   - adjacent direct-copy/layout bucket;
   - projection/GEMM bucket;
   - NCCL bucket;
   - MoE routed/backend bucket;
   - graph replay/eager decode.

9. Write final README.

   Include:

   - exact commands;
   - vLLM source parity table;
   - implemented toggle/variant;
   - focused HC microbench results;
   - correctness results;
   - macro results;
   - final nsys bucket comparison;
   - direct-copy owner comparison for HC owners;
   - promote / keep opt-in / reject decision;
   - next target recommendation.

## Gates

Correctness gate:

- TP8 text smoke passes;
- graph replay remains active;
- eager decode remains `0`;
- HC outputs match current promoted path within a documented tolerance;
- no unexplained text quality regression.

Profile gate:

- reduce 4096/128 rank0 `HC/elementwise + adjacent HC-owned layout` by at
  least `0.15s`; or
- reduce the full HC/elementwise bucket by at least `25%` without increasing
  projection/GEMM, NCCL, or MoE routed/backend enough to cancel the gain.

Macro gate:

- 4096/128/batch4 does not regress by more than `1%`;
- 4096/1024/batch4 improves by at least `3%` for promotion;
- if profile improves but macro is below `3%`, keep the path opt-in and stop.

Scope gate:

- no projection/GEMM backend changes;
- no communication contract changes;
- no precision route changes;
- no sparse attention rewrite;
- no MoE runner finalization or shared-expert overlap work;
- do not promote 07.64 metadata deforestation.

## Stop Conditions

Stop and write the report when:

- one HC opt-in implementation clears correctness, profile, and macro gates;
- one HC implementation improves profile but misses macro promotion;
- vLLM parity review proves mini already has equivalent HC boundaries and the
  bucket is not cleanly reducible;
- the next promising change is projection/GEMM backend work, NCCL, precision,
  sparse attention, or broad runtime ownership;
- two focused HC cuts each produce less than `1%` macro gain and less than
  `0.05s` profile reduction.

Do not keep polishing scattered elementwise nodes after the HC boundary has
been tested.  Select the next target from the final profile.


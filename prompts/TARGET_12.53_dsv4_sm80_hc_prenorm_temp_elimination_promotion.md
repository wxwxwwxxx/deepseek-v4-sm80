# TARGET 12.53: DSV4 SM80 HC Prenorm Temporary Elimination And Promotion Gate

## Background

TARGET 12.49 reran the release long-context and large-batch soak after TARGET
12.52 promoted SWA independent lifecycle into the true no-env release default:

```text
performance_milestones/target12_release_long_context_large_batch_soak/README.md
```

The true release-default path is healthy for text smoke, default-bucket
large-batch decode through batch 128, and the 8192-token long-context rung.
However, the soak exposed a clear prefill memory cliff:

```text
long context: prompt_len=32768, decode_len=16, batch=1 -> OOM
large batch:  prompt_len=128,   decode_len=64, batch=256 -> OOM
```

These two shapes both have 32768 prefill tokens.  Both fail after graph capture
succeeds, during the first prefill model forward, and both point to:

```text
python/minisgl/models/deepseek_v4.py
  -> DeepseekV4DecoderLayer._hc_pre(...)
python/minisgl/kernel/deepseek_v4.py
  -> hc_pre_fallback(...)
  -> flat_float.square().mean(...)
```

The current release default likely enables the older HC Triton split/post path
through the A100 victory bundle:

```text
MINISGL_DSV4_SM80_HC=1
```

but it does **not** enable the newer prenorm cleanup path:

```text
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
```

The default `hc_pre_fallback` therefore still materializes a large FP32
temporary:

```python
flat = x.flatten(1)
flat_float = flat.float()
rsqrt = torch.rsqrt(flat_float.square().mean(-1, keepdim=True) + norm_eps)
mixes = linear_bf16_fp32_fallback(flat, fn) * rsqrt
```

For roughly `32768 * 8 * 2048` elements, the FP32 materialization is about
2 GiB.  That matches the OOM allocation reported by TARGET 12.49.

This target should prove whether the existing fused HC prenorm path eliminates
that temporary safely, and either promote it or identify the exact remaining
work.

## Goal

Eliminate the HC prenorm 2 GiB temporary allocation cliff for large prefill
token counts, while preserving exact-path correctness and the current release
default behavior.

Primary promotion candidate:

```text
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
```

If the existing path passes, promote it into the DSV4 A100/sm80 release default.

If it does not pass, produce a precise blocker and the smallest follow-up
implementation plan:

- fix/extend mini's existing Triton `hc_prenorm_split_pre` path;
- port/adapt vLLM's `mhc_pre` / `mhc_post` boundary;
- or introduce a preallocated workspace only for an unavoidable temporary.

Do not use workspace as the first solution if fusion can remove the temporary.
Avoiding the large intermediate is better than preallocating it.

## Current Code Surfaces

Mini:

```text
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/engine/engine.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
tests/kernel/test_deepseek_v4_wrappers.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
tests/benchmark/test_deepseek_v4_text_smoke.py
tests/engine/test_dsv4_release_defaults.py
```

Relevant mini functions:

```text
dsv4_kernel.hc_pre_fallback
dsv4_kernel.hc_post_fallback
triton.deepseek_v4.hc_prenorm_split_pre
triton.deepseek_v4.hc_split_pre
triton.deepseek_v4.hc_post
```

Historical prompt:

```text
prompts/archive/target07/TARGET_07.68_dsv4_sm80_hc_elementwise_graph_cleanup.md
```

vLLM reference:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/mhc.py
```

SGLang reference, if useful:

```text
/workspace/sglang-main
```

## Required Investigation

### 1. Confirm The Default HC Boundary

Document what the current true release default actually enables:

```text
MINISGL_DSV4_SM80_HC
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP
```

Confirm from runtime logs or active toggles that release default currently
uses HC but not HC_GRAPH_CLEANUP.

### 2. Micro Reproduce The 2 GiB Temporary

Build a focused micro or one-layer harness that calls `hc_pre_fallback` with a
realistic DSV4 shape equivalent to the failing prefill token count:

```text
tokens ~= 32768
hc_mult = real model hc_mult
hidden = real model hidden
dtype = bf16
```

Compare:

```text
default HC path
HC_GRAPH_CLEANUP=1 path
```

Record:

- peak allocated/reserved memory;
- largest single allocation if available;
- wall time;
- output max/mean error versus default path on smaller safe shapes;
- whether `hc_prenorm_split_pre(...)` returns non-None for real shapes.

If a full `32768` micro is too expensive, first use smaller scales that show
linear memory growth, then run the real cliff shape once.

### 3. Correctness Oracle

For safe smaller shapes, compare:

```text
default path
vs
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
```

Check at least:

- `y`;
- `post`;
- `comb`;
- one full decoder layer smoke if practical;
- text sanity with TP8 release default plus HC_GRAPH_CLEANUP.

Define tolerances explicitly.  The path uses FP32 reductions and sigmoid/
softmax/Sinkhorn-style math, so require tight but realistic tolerance rather
than bit-exactness if operation order differs.

### 4. Runtime A/B On The Exact Failure Shapes

Run fresh `torchrun` processes:

#### Default Control

Use the failing shapes from TARGET 12.49:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 32768 \
  --decode-len 16 \
  --batch-size 1 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_53_long32768_default \
  --keep-going
```

Then batch 256:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 128 \
  --decode-len 64 \
  --batch-size 256 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_53_bs256_default \
  --keep-going
```

#### HC Cleanup Candidate

Use a named opt-in env or benchmark variant.  If the existing harness has a
variant for HC graph cleanup, use it; otherwise export the env explicitly and
record that this is an A/B opt-in:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 32768 \
  --decode-len 16 \
  --batch-size 1 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_53_long32768_hc_cleanup \
  --keep-going
```

and:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --prompt-len 128 \
  --decode-len 64 \
  --batch-size 256 \
  --repeats 1 \
  --warmup-repeats 0 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_53_bs256_hc_cleanup \
  --keep-going
```

Required signal:

```text
The 2 GiB allocation from flat.float().square().mean disappears, or the owner
moves to a clearly smaller remaining allocation.
```

### 5. Macro Regression Gate

If the candidate fixes the memory cliff or materially reduces the temporary,
run a normal macro smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
  --num-pages 0 \
  --keep-going \
  --output-dir /tmp/dsv4_target12_53_hc_cleanup_macro
```

Compare against TARGET 12.52 true release-default macro:

```text
historical_4096_128_bs4:      52.611 output tok/s
historical_4096_1024_bs4:     141.863 output tok/s
serving_mixed_112req_wave16:  171.736 output tok/s
prefix_multi_112req_wave16:   113.486 output tok/s
```

Promotion criteria:

- no text sanity regression;
- no CUDA graph replay/eager regression for the default bucket scenarios;
- no material repeatable macro throughput regression;
- long32768 and/or bs256 cliff is fixed or significantly improved;
- no new graph capture/private-pool blowup;
- HC oracle deltas are within documented tolerances.

## Optional Implementation Work

If `HC_GRAPH_CLEANUP=1` does not work out of the box:

1. Find why `hc_prenorm_split_pre(...)` returns `None` for real shapes.
2. Fix only the narrow blocker if it is shape/contiguity/dtype related.
3. If the kernel itself is incorrect or too slow, compare to vLLM's MHC code
   and decide whether to port/adapt it.
4. If a large temporary is unavoidable, design a preallocated HC workspace
   surface and account for it in KV capacity planning.  This is the fallback
   route, not the first choice.

Do not build a broad global workspace manager in this target.  Leave that as a
future release-engine cleanup after the HC owner is understood.

## Required Validation

Minimum static/unit:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kernel/triton/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/engine/test_dsv4_release_defaults.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

TP8 text smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_53_hc_cleanup_text_smoke.json
```

## Output

Write the report to:

```text
performance_milestones/target12_hc_prenorm_temp_elimination/README.md
```

The report must include:

- whether release default currently enables HC and not HC_GRAPH_CLEANUP;
- microbench or one-layer evidence for the 2 GiB temporary;
- A/B result for `HC_GRAPH_CLEANUP=1`;
- correctness oracle deltas;
- TP8 text sanity result;
- long32768 result;
- bs256 result;
- four-scenario macro result if promotion is plausible;
- capacity and memory ledger, including graph private pool and any remaining
  activation/workspace headroom;
- decision: promote HC_GRAPH_CLEANUP, keep opt-in with blocker, or implement a
  narrower HC/vLLM/workspace follow-up.

## Stop Conditions

Stop and report if:

1. `HC_GRAPH_CLEANUP=1` fixes the cliff and passes promotion gates.
2. `HC_GRAPH_CLEANUP=1` fails correctness; report the first mismatching HC
   boundary and do not promote.
3. The kernel does not run for real shapes; report the shape/dtype/contiguity
   blocker and smallest fix.
4. The 2 GiB temporary disappears but another owner becomes the new OOM; record
   the new owner and memory ledger.
5. The path passes memory but causes a material repeatable macro regression.

Do not expand CUDA graph bucket defaults, low precision, MTP, or 1M-context
work inside this target.  Those follow after the HC prefill cliff is understood.

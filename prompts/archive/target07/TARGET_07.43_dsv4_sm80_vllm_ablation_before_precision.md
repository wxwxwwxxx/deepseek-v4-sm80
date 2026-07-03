# TARGET 07.43: DSV4 SM80 vLLM Ablation Before Precision

## Goal

Quantify the value of vLLM core mechanisms before opening the mini-sglang
opt-in FP8 cache/indexer precision lane.

TARGET 07.42 concluded that exact mini-side metadata/runtime microcuts are not
currently justified and recommended TARGET 07.50.  This target inserts one
short evidence step before 07.50: run destructive, env-gated vLLM ablations to
measure whether vLLM's remaining advantage comes from independent engine
mechanisms such as aux-stream overlap, persistent topk/indexer behavior, or
CUDA graph dispatch.

This target should not implement mini-sglang optimizations.  Its output is a
decision: adapt a vLLM engine mechanism first, or proceed to TARGET 07.50.

## Required Input

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.42_dsv4_sm80_vllm_metadata_runtime_parity.md`
- `performance_milestones/target07_vllm_metadata_runtime_parity/README.md`
- `performance_milestones/target07_vllm_metadata_runtime_parity/summaries/vllm_node_trace_attempt.json`
- `performance_milestones/vllm/scripts/run_vllm_matrix.sh`
- `performance_milestones/vllm/scripts/run_vllm_deepseek_v4_matrix.py`

Relevant vLLM code:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/utils/multi_stream_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/cudagraph_dispatcher.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`

Known facts:

- current best mini exact 4096/1024/bs4: `68.8097 output tok/s`;
- fresh vLLM 4096/1024/bs4 reference: about `201.99 output tok/s`;
- fresh vLLM 4096/128/bs4 reference: about `80-82 output tok/s`;
- vLLM selected `deepseek_v4_fp8`, `fp8_ds_mla` KV cache, FP8 indexer cache,
  MXFP4 Marlin MoE, and CUDA graph capture sizes `[1, 2, 4]`;
- vLLM DeepSeek V4 attention currently asserts an FP8 KV cache format, so
  changing vLLM to mini's bf16 flat cache is out of scope for this target.

## Safety And Reproducibility

vLLM source changes are allowed only if they are:

- env-gated;
- minimal;
- reversible;
- recorded as a patch under the milestone directory;
- clearly separated by ablation knob.

The vLLM checkout at `/workspace/vllm-dsv4-docker` is intentionally available
for ablation work.  It currently has a dedicated branch for this project
(`minisgl_docker` at the time this target was written).  It is OK to use normal
git hygiene there to make experiments easier:

- create a temporary ablation branch if useful;
- use `git stash` to move between ablation variants;
- use `git diff` to save exact patches;
- use non-interactive git commands where possible.

Do not clean or modify unrelated untracked ncu report directories in the vLLM
checkout.  Known examples are:

- `/workspace/vllm-dsv4-docker/benchmarks/kernels/mqa_logits_triton_variants/ncu_reports/`
- `/workspace/vllm-dsv4-docker/benchmarks/kernels/ncu_reports/`

Before editing `/workspace/vllm-dsv4-docker`, record:

```bash
git -C /workspace/vllm-dsv4-docker status --short
```

After editing, save the diff:

```bash
git -C /workspace/vllm-dsv4-docker diff \
  > performance_milestones/target07_vllm_ablation_before_precision/summaries/vllm_ablation_patch.diff
```

Do not leave unexplained vLLM worker processes running after failed runs.  If a
run is interrupted, clean up only the vLLM worker/engine processes created by
that run.

## Workloads

Use the same page/block size and TP setup as prior vLLM runs:

- TP8;
- block size `256`;
- `/models/DeepSeek-V4-Flash`;
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`;
- `--max-num-batched-tokens 4096`;
- `--cudagraph-capture-sizes 1,2,4`;
- `--max-cudagraph-capture-size 4`;
- chunked prefill enabled unless the existing script default changes.

Primary workload:

- prompt len `4096`;
- decode len `128`;
- batch size `4`;
- at least `3` measured repeats if runtime is stable.

Confirmation workload:

- prompt len `4096`;
- decode len `1024`;
- batch size `4`;
- run only for the control and ablations that move 4096/128 by at least `5%`,
  or for ablations with surprising/inconclusive 4096/128 behavior.

Example command shape:

```bash
MILESTONE_DIR=/workspace/mini-sglang/performance_milestones/vllm \
OUTPUT_DIR=/tmp/dsv4_vllm_ablation_control_4096x128_bs4 \
performance_milestones/vllm/scripts/run_vllm_matrix.sh \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 3 \
  --warmup-repeats 1 \
  --max-num-batched-tokens 4096 \
  --cudagraph-capture-sizes 1,2,4 \
  --max-cudagraph-capture-size 4
```

## Required Ablations

### 1. Control

Run unmodified vLLM first.  If the control throughput differs from the prior
vLLM line by more than `5%`, rerun once before interpreting ablations.

Record:

- per-repeat output tok/s;
- mean, median, and relative standard deviation;
- output directory;
- vLLM git status and commit.

### 2. DSV4 Aux-Stream Overlap Ablation

Question: how much of vLLM's advantage comes from overlapping indexer with
KV insert/compressor work?

Use an env gate such as:

```text
VLLM_DSV4_ABLATE_AUX_STREAM=1
```

Acceptable implementation choices:

- in `deepseek_v4_attention.py`, pass `None` instead of `self.aux_stream` to
  `maybe_execute_in_parallel` when the env var is set;
- or in `deepseek_v4.py`, avoid creating/passing the `AuxStreamType.Attention`
  stream when the env var is set.

This is the cleanest destructive experiment because
`maybe_execute_in_parallel(..., aux_stream=None)` already means sequential
execution on the current stream.

Decision:

- if output tok/s drops by at least `5%` on 4096/128 and the 4096/1024
  confirmation also drops, plan a mini exact-bf16 aux-stream/custom-op boundary
  adaptation before TARGET 07.50;
- otherwise do not prioritize stream overlap.

### 3. Persistent Topk / Indexer Fast-Path Ablation

Question: how much of vLLM's advantage comes from the FP8 indexer plus
persistent topk path?

Use an env gate such as:

```text
VLLM_DSV4_ABLATE_PERSISTENT_TOPK=1
```

Acceptable implementation:

- in `sparse_attn_indexer.py`, bypass `torch.ops._C.persistent_topk` when the
  env var is set and use the existing non-persistent fallback path if it is
  correct for the shape;
- if the fallback is not valid or OOMs, record the failure and do not spend the
  thread rewriting vLLM topk.

Decision:

- if disabling persistent topk/indexer fast path drops vLLM by at least `5%`,
  decide whether the portable part is exact-bf16 persistent topk workspace or
  whether the value is tied to FP8 paged logits/indexer cache;
- if the value is tied to FP8 cache/indexer layout, proceed to TARGET 07.50.

### 4. CUDA Graph Sanity Ablation

Question: is vLLM's full/piecewise graph dispatch itself a large macro factor
for this workload?

Use the existing benchmark option:

```text
--enforce-eager
```

This is a sanity check, not a direct mini action item.  mini already uses decode
CUDA graph replay, so a large vLLM eager regression does not automatically
imply a new mini optimization.

Decision:

- if eager vLLM drops heavily, record it as supporting evidence that graph
  replay is mandatory;
- only propose mini graph work if the ablation identifies a specific vLLM graph
  mechanism missing from mini, not merely "graphs are good".

## Optional Ablations

Run these only if the required ablations are stable and cheap:

- async output-copy stream ablation;
- custom all-reduce disablement if there is a command-line flag and the run is
  stable;
- `nsys` around one ablation only if macro deltas are large and a profile can
  export reliably.

## Explicitly Out Of Scope

- changing vLLM DeepSeek V4 to bf16 flat KV/indexer cache;
- porting packed FP8 cache/indexer code into mini;
- mini-sglang code changes;
- broad vLLM refactors;
- chasing ablations that crash or OOM after one focused fix attempt;
- treating CUDA graph eager regression as a precise explanation for mini's
  remaining gap.

## Expected Output

Create:

- `performance_milestones/target07_vllm_ablation_before_precision/README.md`
- `raw/`, `scripts/`, and `summaries/` as needed.

The README must include:

| Experiment | Env/patch | 4096/128 mean tok/s | Delta vs control | 4096/1024 tok/s if run | Interpretation | Next decision |
| --- | --- | ---: | ---: | ---: | --- | --- |

Also include:

- exact vLLM patch/diff path;
- control stability summary;
- failed ablations and why they failed;
- whether the evidence supports an exact-bf16 mini adaptation before 07.50;
- whether TARGET 07.50 should start immediately.

## Stop Rules

Stop when one of these is true:

- aux-stream ablation proves a stable `>=5%` vLLM macro loss and therefore
  justifies a mini exact-bf16 overlap/custom-op adaptation target;
- persistent topk/indexer ablation proves a stable `>=5%` vLLM macro loss and
  classifies the mechanism as exact-portable or precision/layout-dependent;
- all required ablations are below `5%` or blocked, so TARGET 07.50 is the next
  best step;
- vLLM source changes become too invasive to remain a clean ablation;
- two consecutive runs are unstable enough that deltas under `5%` cannot be
  interpreted.

## Final Decision Template

End the README with:

- `Decision: run exact-bf16 aux/custom-op adaptation before 07.50`, or
- `Decision: run exact-bf16 persistent-topk/indexer adaptation before 07.50`,
  or
- `Decision: start TARGET 07.50`, or
- `Decision: repeat fair vLLM control because ablation evidence is unstable`.

# TARGET 07.53: DSV4 SM80 Post-FP8-Indexer Reprofile And vLLM Gap Re-ranking

## Goal

Reprofile mini-sglang after the successful vLLM-aligned FP8 indexer backend
port, then re-rank the remaining mini-vs-vLLM performance gap before choosing
the next implementation target.

TARGET 07.52 changed the system enough that older post-splitK profiles are no
longer sufficient.  This target should not start by optimizing a guessed
bottleneck.  It should first answer:

1. What are the top decode graph buckets after FP8 indexer is enabled?
2. Which buckets still differ materially from vLLM's DeepSeek V4 SM80 path?
3. Is the next high-value target graph/layout cleanup, real fused
   `fp8_ds_mla` KV-cache store/gather, projection/GEMM, communication, or
   something else?
4. What minimum PoC should the next target run, and what would prove that it is
   worth continuing?

This is an evidence target.  It may add profiler scripts, classifiers, and
small instrumentation, but it should not do broad kernel implementation unless
one tiny probe is needed to classify a bucket.

## Current State

Best exact baseline before precision:

- 4096/128/batch4: `38.94 output tok/s`;
- 4096/1024/batch4: `68.81 output tok/s`;
- stack: Marlin WNA16 MoE, global topk/lens, bf16 split-K sparse decode,
  decode CUDA graph replay, page size 256, `--num-pages 128`.

TARGET 07.52 opt-in FP8 indexer result:

- microbench large shape batch16/history4096:
  - mini FP8 paged logits `0.1845 ms`;
  - vLLM isolated FP8 paged logits `0.1529 ms`;
  - mini BF16 logits `0.3516 ms`;
  - mini FP8 select `0.2472 ms`, mini BF16 select `0.3709 ms`;
- text smoke passed with CUDA graph replay;
- 4096/128/batch4 FP8 indexer: `41.63 output tok/s`;
- 4096/1024/batch4 FP8 indexer: `73.67 output tok/s`.

Important interpretation:

- FP8 indexer backend port succeeded and should be kept as an opt-in path.
- Relative to the historical best exact path, the macro gain is about `7%`,
  not enough to explain the full vLLM gap.
- For 4096/1024/batch4, mini FP8 indexer still trails:
  - old serving victory line: `114.07 output tok/s`;
  - fresh vLLM reference: about `202 output tok/s`.
- 4096/1024 mini FP8 indexer time is decode dominated:
  - total `55.60 s`;
  - decode forward `47.46 s`;
  - prefill forward `5.47 s`.

Therefore, the next bottleneck is most likely inside repeated decode graph
work, not the isolated indexer logits backend.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.40_dsv4_sm80_post_splitk_reprofile.md`
- `prompts/TARGET_07.43_dsv4_sm80_vllm_ablation_before_precision.md`
- `prompts/TARGET_07.51_dsv4_sm80_vllm_fp8_backend_parity.md`
- `prompts/TARGET_07.52_dsv4_sm80_vllm_fp8_indexer_backend_port.md`
- `performance_milestones/target07_post_splitk_reprofile/README.md`
- `performance_milestones/target07_vllm_ablation_before_precision/README.md`
- `performance_milestones/target07_vllm_fp8_backend_parity/README.md`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/README.md`

Useful existing raw/profile artifacts:

- `performance_milestones/target07_post_splitk_reprofile/summaries/`
- `performance_milestones/target07_vllm_fp8_indexer_backend_port/raw/`
- `performance_milestones/vllm/raw/dsv4_vllm_ablation_control_4096x128_bs4`
- `performance_milestones/vllm/raw/dsv4_vllm_ablation_control_4096x1024_bs4`
- `performance_milestones/vllm/raw/nsys_vllm_4096x128_bs4.sqlite`

Mini code likely relevant for bucket attribution:

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `python/minisgl/engine/`
- `python/minisgl/attention/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/kernel/triton/deepseek_v4.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`

vLLM comparison code:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_compressor.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/`
- `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/mqa_logits_triton.py`
- `/workspace/vllm-dsv4-docker/vllm/compilation/`

## Work Plan

### 1. Preserve 07.52 As The New Baseline

Summarize TARGET 07.52 in the new README:

- microbench wins and quality;
- graph/text smoke status;
- 4096/128 and 4096/1024 macro numbers;
- note that exact-vs-FP8 two-variant runs are invalid when a single engine
  graph is captured at init, so exact and FP8 variants must be separate
  processes for fair graph semantics.

Do not continue optimizing the old mini-owned FP8 indexer slice.

### 2. Run A Fair Short Reprofile Matrix

Run separate single-variant processes for:

- current best exact stack;
- current FP8 indexer stack;
- optionally vLLM control if existing artifacts are stale or not comparable.

Use:

- TP8, 8x A100 sm80;
- model `/models/DeepSeek-V4-Flash`;
- page/block size 256;
- `--num-pages 128` for mini if required to avoid allocation/OOM drift;
- batch size 4;
- prompt length 4096;
- decode length 128 for profiling;
- decode length 1024 for the official long macro if short profile is stable.

Suggested mini FP8-indexer short run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants v1_moe \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 128 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target0753_fp8_indexer_4096x128_bs4_np128 \
  --keep-going
```

Use the actual FP8-indexer promoted variant name if it is not plain `v1_moe`.

### 3. Capture Mini Nsight Profile With FP8 Indexer Enabled

Capture a short 4096/128/batch4 profile for mini's FP8-indexer path.

The profile should be suitable for rank0 SQLite classification:

- include CUDA and NVTX;
- include CUDA graph node information if the installed Nsight supports it;
- keep sample/backtrace disabled unless required;
- keep the workload short enough to avoid multi-hour traces.

Output should be linked or copied under:

```text
performance_milestones/target07_post_fp8_indexer_reprofile/raw/
```

Required outputs:

- mini FP8 indexer benchmark output directory;
- `.nsys-rep` for rank0 or focused rank;
- exported `.sqlite`;
- command script used to capture.

Optional: capture a fresh vLLM 4096/128/batch4 profile only if the existing
vLLM profile is missing, incompatible, or cannot answer the comparison
questions.  Existing vLLM artifacts may be reused, but the README must state
whether they are reused or refreshed.

### 4. Classify The New Mini Profile

Reuse or extend the classifier style from TARGET 07.40.  Classify at least:

- repeat total;
- prefill forward;
- decode forward envelope;
- graph node or kernel buckets when available.

Minimum bucket taxonomy:

| Bucket | Examples |
| --- | --- |
| graph/runtime/copy/cat/index | `copy`, `cat`, `index`, `gather`, `fill`, staging, graph-node data movement |
| elementwise graph nodes | activations, small reductions, normalization/staging helpers |
| FP8 indexer | Q fold, FP8 indexer store, paged logits, topk transform/select |
| sparse attention decode | split-K gather/split/combine, SWA/C4/C128 decode kernels |
| prefill sparse attention | legacy prefill/extend sparse kernels |
| KV/compressor/cache store | compress, norm, RoPE, cache write, packed/bf16 store |
| projection GEMM | Q/WQB/WO_B/selective FP8 or dense projection GEMM |
| MoE/Marlin | routed expert kernel, route prep, expert output combine |
| NCCL/communication | all-reduce, all-gather, communication overlap if visible |
| sampling/logits | lm head, sampling, greedy graph nodes |
| unknown | only when classification is genuinely unclear |

For each bucket report:

- total kernel or node time;
- share of repeat wall time;
- share of decode-envelope time;
- count;
- top representative kernel/node names;
- whether it also appears in vLLM's analogous path.

### 5. vLLM Path Comparison

This target must compare against vLLM explicitly.  Build a mini-vs-vLLM table
for the remaining top buckets.

Required comparisons:

- **Graph/runtime boundary**:
  mini decode CUDA graph replay vs vLLM compile/CUDA graph path.  State whether
  mini still has many copy/cat/index/elementwise nodes inside replay that vLLM
  likely fuses, compiles, or avoids through custom-op boundaries.
- **Indexer**:
  mini FP8 indexer backend after 07.52 vs vLLM FP8 indexer backend from 07.51.
  Confirm whether indexer is no longer a top remaining bucket.
- **KV/cache path**:
  mini still stores main DSV4 attention/cache state mostly as BF16, while vLLM
  uses `deepseek_v4_fp8` and packed `fp8_ds_mla` in the fast path.  If KV/cache
  store or gather is top-two, identify the exact vLLM fused compressor/insert
  path to study next.  Do not recommend standalone
  `quantize_and_insert_k_cache` as the port target.
- **Attention decode**:
  mini bf16 split-K sparse decode vs vLLM sparse/MLA decode boundary.  Confirm
  whether decode split-K remains small or has become top-two.
- **Projection/GEMM and MoE**:
  compare mini's projection and Marlin WNA16 route against vLLM's corresponding
  fused/MXFP4/Marlin path at a source-level boundary.  Do not assume MoE is
  solved if the new profile makes it top-two.
- **Communication**:
  compare all-reduce/all-gather count and bytes.  If communication is top-two,
  check whether vLLM has custom all-reduce or graph-captured communication
  behavior worth adapting.

Use existing vLLM profile and ablation artifacts where possible:

- vLLM control 4096/128/batch4: about `82.28 output tok/s`;
- vLLM control 4096/1024/batch4: about `202.03 output tok/s`;
- aux-stream and persistent-topk ablations were not meaningful macro factors;
- eager ablation was a huge loss but mini already has decode graph replay, so
  the actionable question is graph contents and boundaries, not merely whether
  graph exists.

### 6. Choose The Next Target

End with a concrete next-target recommendation.  Allowed outcomes:

- `Decision: start graph/layout replay deforestation target`
  if runtime/copy/cat/index plus elementwise graph nodes are top-two.
- `Decision: start fused fp8_ds_mla KV-cache store/gather target`
  if KV/cache store, attention gather/dequant, or main attention memory traffic
  is top-two and vLLM's packed KV path plausibly explains the gap.
- `Decision: start projection/GEMM target`
  if projection or dense GEMM is top-two and has a clear vLLM source-level
  backend to adapt.
- `Decision: start MoE/Marlin revisit target`
  if MoE/Marlin has become top-two after indexer improvement.
- `Decision: start communication target`
  if NCCL/all-reduce is top-two and differs materially from vLLM.
- `Decision: blocked; need fresh vLLM profile`
  only if existing vLLM artifacts cannot support the required comparison.

The decision must include:

- expected E2E gain estimate;
- why the chosen target should beat the alternatives;
- do-not-continue condition for the next target.

## Stop Rules

Stop this target when the new profile and vLLM comparison identify the next
primary bottleneck.  Do not spend time polishing low-rank buckets.

Hard stop conditions:

- the top-two buckets are identified with clear evidence;
- mini FP8 indexer path is confirmed to be correctly captured and text-smoke
  safe;
- the profile shows no single bucket with at least `5%` expected E2E upside,
  in which case recommend a fresh vLLM profile or a broader methodology target;
- any attempted instrumentation changes threaten correctness or graph capture.

## Expected Output

Create:

- `performance_milestones/target07_post_fp8_indexer_reprofile/README.md`
- `performance_milestones/target07_post_fp8_indexer_reprofile/scripts/`
- `performance_milestones/target07_post_fp8_indexer_reprofile/raw/`
- `performance_milestones/target07_post_fp8_indexer_reprofile/summaries/`

The README must include:

- copied summary of 07.52 baseline;
- exact commands used;
- mini FP8-indexer macro table;
- vLLM comparison table;
- new nsys/profile classification table;
- top-two bottleneck ranking;
- final next-target decision;
- do-not-continue condition.


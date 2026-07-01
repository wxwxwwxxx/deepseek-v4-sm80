# TARGET 07.392: DSV4 SM80 Post-Marlin Reprofile

## Goal

After TARGET 07.391, establish the new post-Marlin bottleneck ranking for
DeepSeek V4 Flash on A100/sm80, then select the next implementation target from
evidence.

This is primarily a profiling and attribution target. Do not start broad
optimization work inside this thread. The purpose is to prevent spending another
long subthread on non-bottleneck polish now that the MoE expert backend has
changed substantially.

## Background

TARGET 07.391 implemented a mini-owned Marlin WNA16 backend:

```bash
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
```

The 4096/1024/batch4 TP8 macro improved to `54.47 output tok/s` with page size
256, `--num-pages 128`, and DSV4 CUDA graph replay. This is a strong exact-path
improvement, but it remains below the old vLLM serving reference
`114.07 output tok/s`.

The short Nsight profile shifted the top kernel time away from grouped FP4 MoE:

- sparse attention: about `2.067 s`
- `_indexer_bf16_logits_kernel`: about `0.922 s`
- `_hc_split_pre_kernel`: about `0.356 s`
- Marlin WNA16 expert kernel: about `0.234 s`

Therefore the next likely bottleneck is attention/indexer/cache or
metadata/runtime overhead, not the Marlin expert GEMM itself. This target should
confirm that with fair measurements and a mini-vs-vLLM comparison.

## Required Inputs

Read first:

- `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- `prompts/TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port.md`
- `performance_milestones/target07_marlin_wna16_csrc_port/README.md`
- `performance_milestones/target07_marlin_wna16_csrc_port/summaries/csrc_port_summary.json`
- `performance_milestones/target07_marlin_wna16_csrc_port/summaries/nsys_marlin_wna16_4096x128_bs4_np128_rank0.md`
- `performance_milestones/target07_subgraph_parity/README.md`
- `performance_milestones/vllm/README.md`
- Existing mini/vLLM scripts under:
  - `performance_milestones/vllm/scripts/`
  - `benchmark/offline/deepseek_v4_perf_matrix.py`

Important local paths:

- mini model: `python/minisgl/models/deepseek_v4.py`
- mini DSV4 wrappers: `python/minisgl/kernel/deepseek_v4.py`
- mini Marlin helper: `python/minisgl/kernel/marlin_wna16.py`
- mini attention: `python/minisgl/attention/deepseek_v4.py`
- vLLM root: `/workspace/vllm-dsv4-docker`
- vLLM venv: `/workspace/venvs/vllm-dsv4`
- vLLM DSV4 model:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- vLLM DSV4 attention:
  `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

## Scope

In scope:

- create `performance_milestones/target07_post_marlin_reprofile/`;
- run or reuse fair mini Marlin WNA16 macro measurements;
- run or reuse fair old-vLLM measurements when feasible;
- capture short Nsight profiles for the 4096/128/batch4 shape;
- compare mini and vLLM at subgraph/category level;
- estimate E2E gain ceilings with Amdahl-style reasoning;
- decide the next focused target.

Out of scope unless needed for observability:

- major kernel rewrites;
- further Marlin expert GEMM optimization;
- activation quantization, FP8 activation, FP4 activation, or INT8 Tensor Core
  precision-lane work;
- making `marlin_wna16` the default backend;
- changing numerical policy to match vLLM.

Small instrumentation scripts are allowed if they produce clearer reports, but
avoid implementation changes that are not necessary to measure the bottleneck.

## Canonical Mini Configuration

Use the strongest exact mini variant from TARGET 07.391:

```bash
MARLIN_VARIANT=v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache
```

Macro command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants "${MARLIN_VARIANT}" \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 1 \
  --page-size 256 \
  --num-pages 128 \
  --output-dir /tmp/dsv4_target07392_marlin_4096x1024_bs4 \
  --keep-going
```

Also run the short profile shape:

```bash
--prompt-len 4096 --decode-len 128 --batch-size 4
```

For Nsight, do not use `-t nccl`; this environment's Nsight accepts
`cuda,nvtx,osrt,cublas`. Export SQLite after capture and summarize kernels,
runtime APIs, memcpy, NCCL kernels by name, CUDA graph events, and NVTX ranges.

## vLLM Comparison

Prefer a fresh vLLM run if the environment is stable. Use:

- `performance_milestones/vllm/scripts/run_vllm_matrix.sh`
- `performance_milestones/vllm/scripts/nsys_vllm_4096x128_bs4.sh`
- `/workspace/venvs/vllm-dsv4`
- `/workspace/vllm-dsv4-docker`

If vLLM cannot be rerun because of environment or OOM constraints, explicitly
record that and compare against the existing fair vLLM artifacts under
`performance_milestones/vllm/raw/`. Do not silently mix incompatible vLLM
settings. Record block size, TP size, batch size, prompt/decode lengths,
chunked-prefill policy, CUDA graph settings, and scheduler limits.

## Work Plan

1. Create `performance_milestones/target07_post_marlin_reprofile/` with
   `README.md`, `scripts/`, `summaries/`, and `raw/`.

2. Normalize all run configurations:
   - TP8 single-node 8x A100 sm80;
   - model `/models/DeepSeek-V4-Flash`;
   - page/block size 256;
   - batch size 4;
   - prompt length 4096;
   - decode lengths 128 and 1024;
   - mini `--num-pages 128` for the Marlin WNA16 backend;
   - graph capture/replay settings;
   - warmup/repeat counts.

3. Re-run or validate mini macro:
   - `4096/128/batch4`;
   - `4096/1024/batch4`;
   - text smoke if any configuration changes from TARGET 07.391;
   - record output tok/s, decode tok/s, prefill tok/s, TTFT, graph replay
     count, unsupported skips, peak memory, and any JIT/prepack effects.

4. Re-run or validate vLLM macro on the same shapes where feasible. If not
   feasible, link existing artifacts and document the mismatch risk.

5. Capture short Nsight profiles for mini Marlin WNA16 and vLLM
   `4096/128/batch4`. Export `.sqlite` and create summaries. Symlink large
   `.nsys-rep` and `.sqlite` files into the milestone raw directory rather than
   copying them.

6. Build a post-Marlin attribution report with at least these categories:
   - sparse attention and attention cache read;
   - indexer logits, top-k, and indexer cache/store;
   - MoE route metadata, W13 Marlin, SwiGLU/clamp/mul, W2 Marlin, route sum;
   - shared experts if visible;
   - HC/RMSNorm/final linear/logits/sampling;
   - NCCL collectives by semantic label if labels are available;
   - CUDA graph replay, graph breaks, runtime sync, launch overhead, memcpy,
     allocation, and Python/PyTorch small-kernel overhead.

7. Compare mini against vLLM at the corresponding code paths:
   - operator boundaries;
   - tensor shapes;
   - precision lane;
   - graph capture behavior;
   - number of kernels/runtime calls;
   - stream usage and overlap;
   - communication count and placement;
   - memory/KV-cache policy.

8. Produce a ranked next-target decision. For each candidate, include:
   - observed contribution to current mini wall time;
   - mini-vs-vLLM gap if known;
   - estimated E2E upside;
   - likely implementation route: port vLLM, adapt vLLM, local rewrite, or
     reject/defer;
   - correctness risk.

## Decision Rules

Use hard thresholds to avoid non-bottleneck polishing:

- A candidate should normally be in the top two mini contributors or be at
  least `1.3x` slower than the corresponding vLLM subgraph before opening an
  implementation target.
- Do not optimize a subgraph if its Amdahl-style maximum E2E gain is below
  `5%`, unless it is required for correctness or observability.
- Do not continue MoE expert-kernel work if Marlin WNA16 remains below about
  `10%` of the measured workload window.
- Do not open TARGET 07.4 precision lanes unless the report shows that the
  remaining gap is dominated by vLLM-only precision behavior rather than
  attention/indexer/cache/runtime structure.

## Expected Outputs

At minimum:

- `performance_milestones/target07_post_marlin_reprofile/README.md`
- raw or symlinked mini macro outputs;
- raw or symlinked vLLM macro/profile outputs, or explicit blocker notes;
- mini Nsight SQLite summary for `4096/128/batch4`;
- vLLM Nsight SQLite summary for `4096/128/batch4` if feasible;
- one machine-readable summary JSON;
- one ranked bottleneck table;
- one explicit next-target recommendation.

The README should end with:

- `next target`;
- `do not continue here unless...`;
- whether MoE hardening is a side quest or a primary bottleneck;
- whether precision lanes remain deferred.

## Stop Conditions

Stop and write the README once the post-Marlin bottleneck ranking and next
target are clear. Do not implement the selected optimization in this target.

Also stop if:

- mini and vLLM cannot be compared fairly after one focused attempt; record the
  exact mismatch and choose the best local next target from mini-only evidence;
- the available Nsight data is too noisy to rank bottlenecks; record the missing
  capture configuration and produce a rerun recipe;
- the selected next step is outside attention/indexer/cache/runtime/MoE
  hardening, in which case justify it with measured contribution and expected
  E2E upside.

## Done Criteria

- Fresh or validated post-Marlin mini numbers are recorded.
- Fresh or validated vLLM comparison numbers are recorded, or a blocker is
  documented.
- The new top bottlenecks are ranked with evidence.
- The report explicitly explains why MoE expert GEMM is or is not still worth
  optimizing.
- The next implementation target is named and scoped tightly enough for a new
  Codex thread.

# TARGET 07.1: DSV4 sm80 Fair Rebench and vLLM Execution Diff

## Goal

Create an apples-to-apples performance and profiler comparison between
mini-sglang V1 MoE and the old vLLM-based DeepSeek V4 Flash path.

This target is complete when a new Codex thread can point to one milestone
folder and answer:

- which framework is faster on the official 4096/1024/batch4 victory line;
- whether the 4096/128 short nsys profile reproduces the same bottleneck shape;
- how mini's execution path differs from vLLM in MoE, communication, CUDA graph,
  prefill scheduling, and small-kernel fragmentation;
- which vLLM implementation ideas should be ported, adapted, or rejected.

## Primary References

- Master target: `prompts/TARGET_07_dsv4_sm80_vllm_gap_closure.md`
- Current mini V1 evidence: `performance_milestones/v1_moe/README.md`
- vLLM scripts and environment: `performance_milestones/vllm/README.md`
- mini benchmark: `benchmark/offline/deepseek_v4_perf_matrix.py`
- vLLM benchmark shim:
  `performance_milestones/vllm/scripts/run_vllm_deepseek_v4_matrix.py`
- Existing nsys scripts:
  `performance_milestones/vllm/scripts/nsys_minisgl_4096x128_bs4.sh`
  and `performance_milestones/vllm/scripts/nsys_vllm_4096x128_bs4.sh`

vLLM source paths to inspect:

- `/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/`
- `/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py`
- `/workspace/vllm-dsv4-docker/vllm/distributed/device_communicators/custom_all_reduce.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

## Current Baseline Facts

Old vLLM serving result provided by the user, 4096/1024/batch4:

- duration: 35.91s
- output throughput: 114.07 tok/s
- total throughput: 570.78 tok/s
- mean TTFT: 123.21ms
- mean TPOT: 15.68ms

mini V1 MoE recorded 4096/1024/batch4:

- duration: 389.80s
- output throughput: 10.51 tok/s
- decode throughput: 11.25 tok/s
- TTFT: 24.26s
- TPOT: 357.32ms

Existing short nsys comparison is not fully fair yet:

- vLLM used warmup=1, chunked prefill with `max_num_batched_tokens=4096`,
  and cudagraph capture sizes `1,2,4`.
- mini used warmup=0 and DSV4 CUDA graph was disabled by
  `python/minisgl/engine/engine.py`.

## Plan

1. Create `performance_milestones/target07_vllm_gap/`.
   - Keep large raw outputs as symlinks under `raw/`.
   - Copy small summaries under `summaries/`.
   - Add a README with exact commands, git commits, env details, and results.

2. Make the benchmark scripts fair.
   - Ensure mini and vLLM both use TP8, page/block size 256, same prompt/output
     lengths, same batch size, warmup=1, repeats=1.
   - For 4096/128 nsys, run mini both with default prefill budget and
     `--max-extend-tokens 4096` to isolate chunked-prefill effects.
   - Keep vLLM `enable_chunked_prefill=True` and
     `max_num_batched_tokens=4096` for the paired nsys run.

3. Capture official macro results.
   - mini V1 MoE 4096/1024/batch4:
     `torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_perf_matrix.py --model-path /models/DeepSeek-V4-Flash --variants v1_moe --scenarios decode_throughput_bs8 --prompt-len 4096 --decode-len 1024 --batch-size 4 --repeats 1 --warmup-repeats 1 --page-size 256 --output-dir /tmp/dsv4_target07_mini_v1_4096x1024_bs4 --keep-going`
   - vLLM 4096/1024/batch4:
     `OUTPUT_DIR=/tmp/dsv4_target07_vllm_4096x1024_bs4 performance_milestones/vllm/scripts/run_vllm_matrix.sh --scenarios decode_throughput_bs8 --prompt-len 4096 --decode-len 1024 --batch-size 4 --repeats 1 --warmup-repeats 1 --max-num-batched-tokens 4096 --enable-chunked-prefill`

4. Capture short nsys profiles.
   - Use 4096/128/batch4 for iteration speed.
   - Export sqlite for both frameworks.
   - Summarize:
     - kernel count;
     - runtime API count;
     - memcpy count;
     - top kernels by summed duration;
     - NCCL/all-reduce count and duration;
     - CUDA graph evidence;
     - NVTX ranges when available.

5. Write an execution-path diff.
   - Compare mini vs vLLM for:
     - MoE route/prepare/fused experts/finalize;
     - shared expert overlap;
     - TP all-reduce/all-gather boundaries;
     - PyNCCL/custom all-reduce support;
     - CUDA graph dispatch/capture policy;
     - sparse attention and chunked prefill behavior;
     - selective `torch.compile` usage.
   - For each finding, classify as:
     - `port`: vLLM design/code is directly useful;
     - `adapt`: use the design but rewrite for mini;
     - `reject`: not suitable for sm80 or mini;
     - `defer`: needs more evidence.

## Done Criteria

- `performance_milestones/target07_vllm_gap/README.md` contains the fair macro
  comparison and profiler summary.
- The 4096/1024/batch4 result clearly states mini's remaining gap to
  114.07 output tok/s.
- The 4096/128 nsys result includes a top-kernel and top-runtime summary for
  both frameworks.
- The execution-path diff identifies the first optimization target for
  TARGET 07.2.

## Non-Goals

- Do not implement CUDA graph, custom all-reduce, or MoE V2 in this target.
- Do not use vLLM's virtualenv for mini-sglang runs.
- Do not treat vLLM's sm80 reference sparse prefill implementation as a port
  candidate unless its OOM-prone materialization is removed.

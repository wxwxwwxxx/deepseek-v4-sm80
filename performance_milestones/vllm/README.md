# vLLM Comparison Milestone

This folder contains scripts for running the old vLLM-based DeepSeek V4 Flash
path with the same synthetic workload shapes used by
`benchmark/offline/deepseek_v4_perf_matrix.py`.

The scripts keep large artifacts out of git by writing raw results under `/tmp`
and creating symlinks in `raw/`.

## Environment

- vLLM source: `/workspace/vllm-dsv4-docker`
- vLLM virtualenv: `/workspace/venvs/vllm-dsv4`
- model: `/models/DeepSeek-V4-Flash`
- tensor parallel size: 8
- vLLM KV block size: 256

The vLLM virtualenv currently uses `torch==2.11.0+cu128`; do not use it for
mini-sglang runs.

## Scripts

Run the full default mini-sglang workload matrix on vLLM:

```bash
performance_milestones/vllm/scripts/run_vllm_matrix.sh
```

Run the old-framework comparison shape:

```bash
OUTPUT_DIR=/tmp/dsv4_vllm_4096x1024_bs4 \
performance_milestones/vllm/scripts/run_vllm_matrix.sh \
  --scenarios decode_throughput_bs8 \
  --prompt-len 4096 \
  --decode-len 1024 \
  --batch-size 4 \
  --repeats 1 \
  --warmup-repeats 0
```

Capture short Nsight Systems profiles for the paired 4096/128/batch4 workload:

```bash
performance_milestones/vllm/scripts/nsys_vllm_4096x128_bs4.sh
performance_milestones/vllm/scripts/nsys_minisgl_4096x128_bs4.sh
```

The nsys scripts create symlinks like:

- `raw/dsv4_nsys_vllm_4096x128_bs4`
- `raw/nsys_vllm_4096x128_bs4.nsys-rep`
- `raw/nsys_vllm_4096x128_bs4.sqlite`
- `raw/dsv4_nsys_minisgl_v1_moe_4096x128_bs4`
- `raw/nsys_minisgl_v1_moe_4096x128_bs4.nsys-rep`
- `raw/nsys_minisgl_v1_moe_4096x128_bs4.sqlite`

## Notes

The vLLM runner uses token-id prompts and `detokenize=False` to match the
synthetic offline mini-sglang benchmark as closely as possible. It reports
throughput and elapsed time. Offline `LLM.generate()` returns completed outputs,
so this runner does not report serving-style TTFT/TPOT; use the nsys traces and
the serving benchmark for fine-grained latency comparisons.

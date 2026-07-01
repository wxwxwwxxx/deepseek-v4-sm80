# TARGET 07.1: DSV4 SM80 vLLM Gap Fair Rebench

Status: TARGET 07.1 fair rebench is complete enough to enter TARGET 07.2.
The measured results and next-path recommendation are in `RESULTS.md`.

## Goal

The TARGET 07 win condition is DeepSeek V4 Flash on TP8, page/block size 256,
4096 input tokens, 1024 output tokens, batch size 4, with mini-sglang exceeding
the old vLLM-based serving baseline:

| System | Shape | Output tok/s | Other |
| --- | --- | ---: | --- |
| old vLLM serving baseline | 4096/1024/bs4 | 114.07 | TTFT 123.21 ms, TPOT 15.68 ms |
| mini v1_moe existing macro | 4096/1024/bs4 | 10.5079 | warmup=0, TTFT 24.256 s |

TARGET 07.1 is not an optimization target. It makes the evidence fair enough
to decide whether TARGET 07.2 should start with communication/CUDA graph work or
whether another rebench is needed first.

## Fairness Contract

All new fair commands use:

- model: `/models/DeepSeek-V4-Flash`
- TP size: 8
- page/block size: 256
- prompt length: 4096
- batch size: 4
- repeats: 1
- warmup repeats: 1
- mini environment: current mini-sglang environment, not the vLLM venv
- vLLM environment: `/workspace/venvs/vllm-dsv4`, source tree
  `/workspace/vllm-dsv4-docker`

For vLLM, the fair path keeps the previously observed production-relevant
settings: chunked prefill with `max_num_batched_tokens=4096`, CUDA graph capture
sizes `1,2,4`, max CUDA graph capture size `4`, custom all-reduce enabled.

For mini, the fair nsys pass must be run twice:

- default prefill scheduling
- `MAX_EXTEND_TOKENS=4096`, to isolate the chunked-prefill fairness variable

DeepSeek V4 CUDA graph is still disabled in mini by code, so the fair rebench
will document the gap rather than masking it.

## Run Order

Macro 4096/1024 fair throughput:

```bash
bash performance_milestones/target07_vllm_gap/scripts/run_mini_4096x1024_bs4.sh
bash performance_milestones/target07_vllm_gap/scripts/run_vllm_4096x1024_bs4.sh
```

Short 4096/128 Nsight Systems fair profiles:

```bash
bash performance_milestones/target07_vllm_gap/scripts/nsys_mini_4096x128_bs4_fair.sh
MAX_EXTEND_TOKENS=4096 bash performance_milestones/target07_vllm_gap/scripts/nsys_mini_4096x128_bs4_fair.sh
bash performance_milestones/target07_vllm_gap/scripts/nsys_vllm_4096x128_bs4_fair.sh
```

The mini nsys script profiles worker rank 0 by default. This avoids a current
container/Nsight/NCCL crash seen when wrapping the `torchrun` launcher itself.
To profile more mini ranks, set `NSYS_PROFILE_RANKS=0,1` or
`NSYS_PROFILE_RANKS=all`; each profiled rank writes a separate
`*_rank<N>.nsys-rep` and sqlite export.

The mini nsys script also uses `--memory-ratio 0.8` by default through
`NSYS_MEMORY_RATIO=0.8`. This leaves Nsight/CUDA/NCCL enough headroom on A100
80GB; the default benchmark memory ratio `0.9` reproduced a rank0 NCCL
segfault under CUDA tracing in this container. The active 4096/128 workload is
unchanged. Set `NSYS_MEMORY_RATIO=0.9` only if you intentionally want to retry
the tighter allocation.

Link the older artifacts and generate sqlite summaries:

```bash
bash performance_milestones/target07_vllm_gap/scripts/link_existing_artifacts.sh
```

All scripts accept `DRY_RUN=1` and pass through extra benchmark flags after the
script arguments.

## Script Outputs

| Script | Primary output | Summary output |
| --- | --- | --- |
| `run_mini_4096x1024_bs4.sh` | `/tmp/dsv4_target07_mini_v1_4096x1024_bs4_warmup1` | copied under `summaries/` |
| `run_vllm_4096x1024_bs4.sh` | `/tmp/dsv4_target07_vllm_4096x1024_bs4_warmup1` | copied under `summaries/` |
| `nsys_mini_4096x128_bs4_fair.sh` | `/tmp/dsv4_target07_nsys_mini_v1_4096x128_bs4_*_warmup1` and `/tmp/nsys_target07_mini_v1_4096x128_bs4_*_warmup1_rank<N>.nsys-rep` | sqlite summary under `summaries/` |
| `nsys_vllm_4096x128_bs4_fair.sh` | `/tmp/dsv4_target07_nsys_vllm_4096x128_bs4_warmup1` | sqlite summary under `summaries/` |
| `summarize_nsys_sqlite.py` | input sqlite | JSON/Markdown profiler digest |

Large historical mini sqlite exports can use `--lite` to skip expensive top-k
kernel grouping while retaining table counts, durations, CUDA graph information,
memcpy totals, and NCCL NVTX counts.

## Existing Evidence Indexed Here

The current directory links or copies the pre-existing artifacts so TARGET 07.1
has one stable place to compare evidence.

| Artifact | Fairness status | Key result |
| --- | --- | --- |
| `existing_mini_v1_moe_4096x1024_bs4_warmup0` | not fair, warmup=0 | 10.5079 output tok/s, 11.2544 decode tok/s |
| `existing_mini_v1_moe_4096x128_bs4_warmup0` | not fair, warmup=0 | 5.0653 output tok/s, 7.0195 decode tok/s |
| `existing_vllm_4096x128_bs4_warmup1` | fair on vLLM side | 80.8257 output tok/s |
| `existing_nsys_mini_v1_moe_4096x128_bs4` | not fair, warmup=0 | 26,833,000 kernels, 30,493,265 runtime calls, no CUDA graph events |
| `existing_nsys_vllm_4096x128_bs4` | fair on vLLM side | 124,480 kernels, 1,908,662 runtime calls, 7,200 CUDA graph events |

The existing mini 4096/128 profile is still useful for the scale of the issue:
it shows no CUDA graph events and orders of magnitude more kernel/runtime calls
than vLLM. It should not be treated as the final fair comparison because it used
`warmup_repeats=0`, while the vLLM profile used warmup and CUDA graph capture.

## Execution Diff

The detailed mini/vLLM path comparison and `port`/`adapt`/`reject`/`defer`
decisions are in `EXECUTION_DIFF.md`.

## Current Judgment

Enter TARGET 07.2 next. The fair 4096/1024 macro run still leaves mini at
10.5768 output tok/s versus vLLM at 201.874 output tok/s and the old serving
baseline at 114.07 output tok/s. The short 4096/128 nsys comparison shows mini
at 5.5071 output tok/s versus vLLM at 80.9050 output tok/s, with no DSV4 CUDA
graph events in mini and far more CUDA/NCCL launches.

The next implementation axis should be communication labeling/PyNCCL or
custom-op graph-readiness, followed by DSV4 decode CUDA graph replay for capture
sizes `1,2,4`. MoE V2 should wait until those deltas are measured.

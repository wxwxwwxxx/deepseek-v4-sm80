# TARGET 07.43: vLLM Ablation Before Precision

## Status

Completed.  No mini-sglang performance path was modified.

The vLLM checkout was changed only with env-gated Python ablation knobs and the
exact diffs were saved under `summaries/`.

## vLLM Source State

- vLLM root: `/workspace/vllm-dsv4-docker`
- Branch: `minisgl_docker`
- Commit: `bfaea783f5192189b49ca21c2893f7266345e09c`
- Initial status: only the two known untracked ncu report directories.
- Final status: the two env-gated source edits plus the same untracked ncu
  report directories.

Patch artifacts:

- `summaries/vllm_aux_stream_ablation_patch.diff`
- `summaries/vllm_persistent_topk_ablation_patch.diff`
- `summaries/vllm_ablation_patch.diff`

Raw metrics:

- `summaries/ablation_metrics.json`

Output directories:

| Experiment | Output directory | Milestone raw link |
| --- | --- | --- |
| Control 4096/128 | `/tmp/dsv4_vllm_ablation_control_4096x128_bs4` | `raw/dsv4_vllm_ablation_control_4096x128_bs4` |
| Aux-stream ablation 4096/128 | `/tmp/dsv4_vllm_ablation_aux_stream_4096x128_bs4` | `raw/dsv4_vllm_ablation_aux_stream_4096x128_bs4` |
| Persistent-topk ablation 4096/128 | `/tmp/dsv4_vllm_ablation_persistent_topk_4096x128_bs4` | `raw/dsv4_vllm_ablation_persistent_topk_4096x128_bs4` |
| Eager default failed 4096/128 | `/tmp/dsv4_vllm_ablation_enforce_eager_4096x128_bs4` | `raw/dsv4_vllm_ablation_enforce_eager_4096x128_bs4_failed_default` |
| Eager mem080 4096/128 | `/tmp/dsv4_vllm_ablation_enforce_eager_4096x128_bs4_mem080` | `raw/dsv4_vllm_ablation_enforce_eager_4096x128_bs4_mem080` |
| Control 4096/1024 | `/tmp/dsv4_vllm_ablation_control_4096x1024_bs4` | `raw/dsv4_vllm_ablation_control_4096x1024_bs4` |
| Eager mem080 4096/1024 | `/tmp/dsv4_vllm_ablation_enforce_eager_4096x1024_bs4_mem080` | `raw/dsv4_vllm_ablation_enforce_eager_4096x1024_bs4_mem080` |

## Results

Primary workload: TP8, block size 256, `/models/DeepSeek-V4-Flash`,
4096 prompt tokens, 128 decode tokens, batch size 4, repeats 3, warmup 1,
chunked prefill enabled, CUDA graph capture sizes `1,2,4`.

| Experiment | Env/patch | 4096/128 mean tok/s | Delta vs control | 4096/1024 tok/s if run | Interpretation | Next decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Control | Unmodified vLLM for 4096/128; env gates unset for 4096/1024 confirmation | 82.2825 | 0.00% | 202.0342 | Stable and matches the prior 80-82 short reference and about 202 long reference. | Use as baseline. |
| DSV4 aux-stream overlap ablation | `VLLM_DSV4_ABLATE_AUX_STREAM=1`; `deepseek_v4_attention.py` passes `aux_stream=None` to `maybe_execute_in_parallel` | 81.8386 | -0.54% | Not run | Disabling the aux-stream overlap did not produce a meaningful macro loss on 4096/128. | Do not prioritize exact-bf16 aux/custom-op adaptation before 07.50. |
| Persistent topk/indexer fast-path ablation | `VLLM_DSV4_ABLATE_PERSISTENT_TOPK=1`; `sparse_attn_indexer.py` bypasses `_C.persistent_topk` and uses existing decode fallback | 82.4560 | +0.21% | Not run | No loss from disabling persistent topk at this macro shape. Any value is not proven as an exact-bf16 standalone target. | Do not start exact-bf16 persistent-topk/indexer adaptation before 07.50. |
| CUDA graph sanity ablation | `--enforce-eager`; default `gpu_memory_utilization=0.9` OOMed, focused retry used `--gpu-memory-utilization 0.80` | 24.8294 | -69.82% | 30.5310 (control 202.0342, -84.89%) | vLLM depends heavily on compile/CUDA graph dispatch for this workload. The default eager run also exposed an OOM in eager sparse prefill reference path. | Record as graph-mandatory evidence only; mini already has decode graph replay, so this does not create a new exact-bf16 action item. |

## Stability

Control 4096/128 per-repeat output tok/s:

- `82.2278`, `82.3471`, `82.2725`
- mean `82.2825`, median `82.2725`, relative stddev `0.0598%`

Ablation per-repeat output tok/s:

- aux-stream off: `81.7781`, `81.8389`, `81.8988`; relative stddev `0.0602%`
- persistent topk off: `82.3266`, `82.6071`, `82.4344`; relative stddev `0.1401%`
- enforce eager mem080: `24.7245`, `24.8789`, `24.8848`; relative stddev `0.2989%`

The aux-stream and persistent-topk deltas are far below the 5% decision bar.
The eager delta is large and was confirmed at 4096/1024.

## Failed/Adjusted Ablation

The first eager run used the standard `gpu_memory_utilization=0.9` and failed
before producing throughput.  It OOMed in
`deepseek_v4_attention.py::_ref_sparse_attn_prefill` at
`kv.index_select(...).reshape(...).float()`, trying to allocate about 5.00 GiB
with about 4.68-4.69 GiB free on multiple ranks.

One focused retry reduced only `--gpu-memory-utilization` to `0.80`, leaving
enough headroom for the eager reference prefill path.  That retry produced the
reported eager throughput.  This adjustment changes KV-cache headroom, not the
core eager-vs-graph mechanism, and is recorded as a sanity ablation rather than
a fair serving configuration.

## Interpretation

The destructive vLLM experiments do not show that the remaining mini-vs-vLLM
gap is primarily due to aux-stream overlap or persistent topk/indexer behavior
as an exact-bf16 portable mechanism.

CUDA graph/compile dispatch is a large vLLM macro factor, but this is not a new
mini action item by itself.  mini already uses decode CUDA graph replay in the
current exact stack, and this ablation did not identify a specific missing
mini-side graph mechanism beyond the broad fact that eager vLLM is much slower.

The strongest evidence-backed remaining gap source is still vLLM's precision
and layout lane: `deepseek_v4_fp8`, packed `fp8_ds_mla` KV cache, FP8 indexer
cache, and paged FP8 logits/indexer behavior.

## Decision

Decision: start TARGET 07.50.

Do not start an exact-bf16 aux/custom-op adaptation before 07.50.
Do not start an exact-bf16 persistent-topk/indexer adaptation before 07.50.

Current best exact mini result remains:

- 4096/128/bs4: `38.9379 output tok/s`
- 4096/1024/bs4: `68.8097 output tok/s`

TARGET 07.50 should remain opt-in and should not change the default exact bf16
policy.

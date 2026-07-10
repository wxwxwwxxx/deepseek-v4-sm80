# TARGET 12.57: DSV4 SM80 Release Fallback Census And Native Backend Gate

## Status

Run after TARGET 12.56.

TARGET 12.56 promoted chunked prefill for the DSV4 A100/sm80 release default
with a conservative `max_extend_tokens=8192` budget.  The important conclusion
was:

```text
chunked prefill + SWA independent lifecycle works
8192-token chunks pass the tested 262k long-context point
larger chunks are blocked by long-context indexer/model fallback temporaries
```

This target should therefore avoid re-litigating scheduler chunking.  Treat
the current release default as the baseline and focus on the remaining real
torch/Python fallback paths that either:

- block larger prefill chunks;
- threaten 512k/1M long-context smoke;
- make large-batch release behavior silently fall back to slow paths.

## Purpose

Audit the current DSV4 A100/sm80 release default for actual runtime fallback
execution, then either remove the highest-impact fallback owner or produce a
ranked native-backend plan.

The first owner to investigate is the long-context indexer fallback family:

```text
python/minisgl/models/deepseek_v4.py:1910
  attn_backend.select_indexer_fp8

python/minisgl/attention/deepseek_v4.py
  _compressed_raw_to_component_locs
  torch.full_like

python/minisgl/kernel/deepseek_v4.py
  indexer_fp8_paged_logits_fallback
  torch.full
```

TARGET 12.56 showed:

```text
65536 / budget 32768:
  OOM in _compressed_raw_to_component_locs -> torch.full_like
  failing allocation around 128 MiB

131072 / budget 24576:
262144 / budget 16384:
  OOM in indexer_fp8_paged_logits_fallback -> torch.full
  failing allocation around 2.25 GiB

262144 / budget 8192:
  pass, 32 prefill chunks, decode graph 1 replay / 0 eager
```

The immediate goal is not to change precision, graph-capture prefill, or decode
graph bucket policy.  The immediate goal is to stop long-context release shapes
from depending on large torch fallback temporaries.

## Non-Goals

- Do not re-enable or modify MTP.
- Do not introduce FP8/INT8 precision changes.
- Do not expand decode CUDA graph buckets in this target.
- Do not add prefill CUDA graph capture.
- Do not chase every function whose name contains `fallback`; classify actual
  runtime backends and prioritize only real release-path fallback.
- Do not make torch.compile the default.  It may be measured as a short-term
  oracle only if it helps attribute an owner.

## References To Read First

Current evidence:

```text
performance_milestones/target12_chunked_prefill_long_context/README.md
prompts/TARGET_12.56_dsv4_sm80_chunked_prefill_long_context.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

Mini source:

```text
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/scheduler/cache.py
python/minisgl/scheduler/prefill.py
benchmark/offline/deepseek_v4_perf_matrix.py
```

SGLang reference:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
```

vLLM reference:

```text
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
```

Prefer adapting mature SGLang/vLLM behavior when the mechanism is clear.  If
mini already has an equivalent Triton/CUDA backend, first determine why the
release path is not dispatching to it.

## Work Items

### 1. Reproduce The 12.56 Baseline

Run a small confirmation sweep using the current release default:

```text
4096 / 1024 / bs4
65536 / 8 / bs1 / serving-default max_extend_tokens
262144 / 1 / bs1 / serving-default max_extend_tokens
```

Record:

```text
effective max_extend_tokens
prefill chunk lengths
decode graph replay/eager counts
peak allocated/reserved memory
text/token sanity
owner timing and backend counters
```

Use `--use-serving-max-extend-tokens` when the goal is to test the true serving
default.  Use explicit `--max-extend-tokens` only for the failure/optimization
sweep.

### 2. Reproduce The Failing Owner Shapes

Re-run the known failing larger-chunk points with owner attribution:

```text
65536 / 8 / bs1 / max_extend_tokens 32768
131072 / 1 / bs1 / max_extend_tokens 24576
262144 / 1 / bs1 / max_extend_tokens 16384
```

For each failure, capture:

```text
stack owner
failing allocation size
tensor shapes around the owner
actual backend selected
whether the Triton/CUDA path was skipped, unsupported, disabled, or failed
```

The report must distinguish:

```text
release default failure
manual larger-chunk failure
performance-only larger-chunk limitation
```

### 3. Classify Actual Runtime Backends

For the release-default and failing shapes, classify hot wrappers by actual
runtime path:

```text
torch/Python fallback
Triton
CUDA extension
cuBLAS
Marlin
in-graph metadata
optional None skip
unsupported skip
```

The naming convention is not enough.  A function named `fallback` may dispatch
to Triton; a function without `fallback` may allocate a large torch temporary.

Produce a table with:

```text
owner / function
actual backend
shape
time or memory owner
release impact
native-backend candidate
```

### 4. Investigate Existing Native Indexer Paths First

Before writing a new kernel, answer:

```text
Does python/minisgl/kernel/triton/deepseek_v4.py provide a usable paged FP8
indexer logits/select path for the failing shapes?

If yes, why does the release path fall back?
  toggle disabled?
  dtype/layout mismatch?
  unsupported width/static_max_seq_len?
  capture/eager guard?
  ABI/import failure?
  correctness guard?

If no, is there a SGLang or vLLM native path we can port or mirror?
```

The target should prefer these fixes, in order:

1. enable or repair an existing mini Triton/CUDA path;
2. adapt SGLang/vLLM's native indexer/metadata behavior;
3. implement a minimal mini-owned Triton/CUDA backend.

### 5. Avoid Full Long-Context Logit Materialization

The key anti-pattern to remove is full materialization of long-context scratch
buffers such as:

```text
torch.full((rows, max_seq_len), -inf, dtype=torch.float32, ...)
torch.full_like(locs, -1)
full dequantize/cache materialization when only top-k/select is needed
```

Preferred backend shape:

```text
stream/gather paged cache blocks
compute scores in blocks
maintain top-k/select state directly
return only selected raw/page/full/component locations
avoid [rows, max_seq_len] FP32 logits unless explicitly requested by a debug oracle
```

If a full logits tensor is still needed for correctness comparison, keep it as
an explicit debug/oracle path, not the release default.

### 6. Validate Correctness And Performance

At minimum, keep the release default healthy:

```text
4096 / 1024 / bs4
65536 / 8 / bs1 / serving default 8192
262144 / 1 / bs1 / serving default 8192
```

If a native/fused indexer fix lands, re-run the larger-chunk gates:

```text
131072 / 1 / bs1 / max_extend_tokens 24576
262144 / 1 / bs1 / max_extend_tokens 16384
```

Optional stretch gates:

```text
262144 / 1 / bs1 / max_extend_tokens 24576
524288 / 1 / bs1 / serving default 8192
1048576 / 1 / bs1 / serving default 8192
```

Treat 512k/1M as smoke tests first, not throughput benchmarks.  If they fail,
record the first owner and whether it is the same indexer family or a new
long-context limit.

Correctness gates:

```text
text sanity: no garbled or obviously malformed output
where feasible, compare chunked vs unchunked token ids on a shorter oracle
focused pytest for cache/chunked prefill/release defaults
```

### 7. Termination Conditions

Stop the target after one of these is true:

- the indexer fallback temporary is fixed or bypassed and the larger-chunk gate
  improves;
- the indexer native path is proven unavailable/high-risk and the report gives
  a concrete next target for porting it;
- 512k/1M smoke finds a different first-order blocker that should supersede
  indexer work;
- the remaining fallback owners are low-impact compared with chunked-prefill
  TTFT and decode graph replay, with evidence.

Do not spend the whole target shaving tiny fallback counters after the main
long-context owner has been classified.

## Output

Write:

```text
performance_milestones/target12_release_fallback_census_native_backend_gate/README.md
```

The report should include:

- baseline confirmation table;
- failing owner reproduction table;
- actual backend census table;
- SGLang/vLLM comparison notes;
- native backend feasibility decision;
- before/after memory and TTFT for any implemented fix;
- clear recommendation:

```text
promote / keep opt-in / defer / open follow-up target
```

If code changes are made, include exact files changed and validation commands.

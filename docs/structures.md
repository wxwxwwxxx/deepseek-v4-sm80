# Mini-SGLang DSV4 structure

This document describes the retained DeepSeek V4 Flash serving architecture in
the `0.1.0+dsv4.sm80` downstream release.

## Processes and request flow

The OpenAI-compatible API server, tokenizer/detokenizer workers, and one
scheduler/engine process per tensor-parallel rank communicate over ZeroMQ for
control traffic. On the validated TP8 deployment, the eight engine ranks use
PyNCCL for optimized tensor-parallel collectives. The explicit fallback oracle
uses the retained reference collective path instead.

1. The API server accepts a chat completion and sends it to the tokenizer.
2. The tokenizer applies the DeepSeek V4 chat template and forwards token IDs
   to scheduler rank 0.
3. Rank 0 broadcasts request state to the other TP ranks.
4. Each scheduler admits work, manages prefix/SWA/component cache state, and
   invokes its local DeepSeek V4 engine shard.
5. Decode uses a captured CUDA graph when the active shape is covered and
   otherwise uses the same optimized runtime eagerly.
6. Rank 0 gathers output tokens; the detokenizer streams text back through the
   API server.

Offline `LLM` uses the same scheduler and engine classes without the HTTP and
tokenizer worker processes. A torchrun process supplies each TP rank's
`DistributedInfo`.

## Retained package ownership

- `minisgl.models.deepseek_v4`: the only model implementation and checkpoint
  sharding/remapping owner.
- `minisgl.attention.deepseek_v4`: C4, C128, indexer, and SWA attention and
  metadata behavior.
- `minisgl.kvcache`: the DeepSeek V4 cache pool, radix prefix manager, and
  reference naive manager.
- `minisgl.engine`: model loading, recipe resolution, CUDA graph policy, and
  per-rank execution.
- `minisgl.scheduler`: request admission, chunked prefill, decode scheduling,
  and cache lifecycle.
- `minisgl.kernel`: DSV4 Triton/CUDA JIT kernels, Marlin WNA16, PyNCCL, and
  radix bindings.
- `minisgl.server`: CLI parsing, process launch, `/v1/models`, and
  `/v1/chat/completions`.
- `minisgl.llm`: the offline Python interface.
- `minisgl.benchmark`: shared online-client measurement helpers.

Unsupported model implementations and selectable generic attention backends
are not present. FlashInfer remains a sampling dependency, not a selectable
attention backend. MTP is not part of the release.

## Public and developer benchmark surfaces

The stable scripts are the two offline entries under `benchmark/offline/` and
the two online entries under `benchmark/online/`. DeepSeek V4 correctness,
microbenchmark, profiling, and performance-matrix harnesses are isolated under
`debug/dsv4/benchmark/offline/` and are not stable public APIs.

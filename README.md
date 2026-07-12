# DeepSeek V4 Flash serving on NVIDIA A100/SM80, based on Mini-SGLang

High-performance **DeepSeek V4 Flash** serving for NVIDIA A100 and other sm80 GPUs,
built as a specialized downstream release of
[Mini-SGLang](https://github.com/sgl-project/mini-sglang).

Many optimized kernels used by modern DeepSeek serving stacks target newer GPU
architectures and are unavailable on sm80. This project adapts Mini-SGLang for
DeepSeek V4 Flash on A100, supplies sm80-compatible kernels and runtime paths,
and applies practical performance tuning for tensor-parallel serving. It also
supports chunked prefill and single-request contexts up to the model's 1M-token
limit on the validated DGX A100 platform.

See the [DGX A100 performance results](PERFORMANCE.md) for measured throughput,
CUDA graph memory tradeoffs, and long-context capacity.

## Highlights

- **DeepSeek V4 Flash on sm80:** a focused implementation for the model's sparse
  attention, indexer, compression, hybrid-computation, and MoE architecture.
- **A100-native MoE:** packed FP4 expert weights run through a ported Marlin
  WNA16 backend with BF16 activations and BF16 tensor-parallel reduction.
- **Optimized projection and attention paths:** cached BF16 projection weights,
  fused Triton/CUDA kernels, and native C4/C128 sparse-attention metadata paths.
- **Serving-oriented runtime:** CUDA graph decode replay, size-aware PyNCCL
  communication, radix prefix caching, independent SWA lifetime, and chunked
  prefill.
- **Model-aligned precision:** BF16 activations and primary compute, while
  preserving the model's FP32 state and quantized FP8/FP4 weights.
- **Long-context support:** 512K and 1M single-sequence capability has been
  validated with page size 256 and bounded prefill chunks.
- **Simple public surface:** an optimized default path and an explicit slow
  fallback/oracle path for diagnosis.

This release serves **DeepSeek V4 Flash only**. The validated platform is one
DGX with **8x NVIDIA A100-SXM4-80GB**, TP8, CUDA 12.8.2, and NCCL 2.26.2-1.
Other sm80 systems may require different memory and CUDA graph settings.

## Install

Install from the current source checkout:

```bash
python -m pip install -e .
```

Or build and install a wheel:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/minisgl-0.1.0+dsv4.sm80-*.whl
```

WildChat benchmarking uses the optional benchmark dependencies:

```bash
python -m pip install -e '.[benchmark]'
```

## Quick Start

Start an OpenAI-compatible server on eight GPUs:

```bash
python -m minisgl \
  --model /models/DeepSeek-V4-Flash \
  --tp-size 8 \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 1919
```

Send a request:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with only 4: 2 + 2 ="}],
    "max_completion_tokens": 16,
    "temperature": 0
  }'
```

The API supports text-only streaming and non-streaming chat completions. It
returns explicit OpenAI-style errors for unsupported options instead of
silently ignoring them. The model ID reported at startup is the ID clients
should use; set it explicitly with `--served-model-name` when needed.

An interactive shell is also available:

```bash
python -m minisgl.shell --model /models/DeepSeek-V4-Flash --tp-size 8
```

## Runtime Settings

Ordinary DGX A100 use requires no DeepSeek-specific tuning flags. The optimized
runtime defaults to page size 256, chunked prefill, up to 256 running requests,
and CUDA graph buckets through M=256.

Optional recipes set the request capacity, maximum captured decode batch, and,
for long-context modes, maximum sequence length:

| Recipe | Max running requests | CUDA graph max M | Context length | Intended use |
| --- | ---: | ---: | ---: | --- |
| `dsv4_sm80_low_m64` | 256 | 64 | Model config | More KV capacity; batches above M=64 run eagerly. |
| `dsv4_sm80_mid_m128` | 256 | 128 | Model config | Capacity/throughput balance through M=128. |
| `dsv4_sm80_balanced` | 256 | 256 | Model config | DGX A100 throughput-oriented default. |
| `dsv4_sm80_long_context_512k` | 4 | 4 | 524,288 | Low-concurrency 512K context serving. |
| `dsv4_sm80_1m_smoke` | 1 | 1 | 1,048,576 | Single-request 1M capability smoke. |

All recipes retain the optimized runtime's page size 256, prefill chunk size
8,192, and memory ratio 0.9 unless explicitly overridden. Select a recipe with
`--recipe NAME`. These settings were measured on a DGX A100 8x80GB
system and are templates rather than universal sm80 defaults. Explicit
command-line settings take precedence over the corresponding recipe fields.

### Performance and memory arguments

Most users should start with the defaults or a recipe. The following options
are the useful controls when adapting them to another workload or sm80 system:

| Argument | What it controls | Main tradeoff |
| --- | --- | --- |
| `--tp-size N` | Number of tensor-parallel GPU workers. | This release is validated with TP8; changing it alters per-GPU weights, cache capacity, and communication. |
| `--max-running-requests N` | Maximum number of simultaneously active request slots. | Higher values allow more concurrency but increase request metadata and independent SWA reservation. |
| `--cuda-graph-max-bs N` | Largest decode batch captured by CUDA Graph. | Larger values cover higher active M but consume more graph memory, reduce KV capacity, and increase startup time. Batches above this value remain legal and run eagerly. |
| `--context-length N` | Maximum prompt plus generated tokens for one sequence, overriding the model config. | Larger values widen request/page tables; actual admission is still limited by available KV capacity. |
| `--memory-ratio R` | Fraction of GPU memory made available to the runtime capacity planner. | Raising it can provide more KV pages but leaves less safety headroom for allocations outside the planned budget. |
| `--max-prefill-length N` | Maximum number of tokens processed by one chunked-prefill forward. | Larger chunks may improve prefill efficiency but increase activation/workspace peaks; smaller chunks reduce peak memory. |

Low-level page-count and page-size overrides are intentionally omitted here;
the release defaults are part of the validated DeepSeek V4 cache layout and
normally should not be changed.

For a slow correctness reference, use the fallback runtime:

```bash
python -m minisgl \
  --model /models/DeepSeek-V4-Flash \
  --tp-size 8 \
  --dsv4-runtime fallback
```

## Benchmarks

The public benchmark scripts are intentionally small and easy to edit:

```bash
python benchmark/offline/bench.py
python benchmark/offline/bench_wildchat.py
python benchmark/online/bench_simple.py
python benchmark/online/bench_qwen.py
```

The online scripts target a running server. `Qwen` in `bench_qwen.py` refers to
the request-trace format, not a supported model. Kernel microbenchmarks,
correctness probes, and profiling harnesses are kept under
[`debug/dsv4/`](debug/dsv4/README.md).

## Current Scope

- DeepSeek V4 Flash is the only supported model.
- MTP speculative decoding is not included.
- The OpenAI-compatible endpoint is text-only and does not claim full OpenAI
  API parity.
- The package version is `minisgl==0.1.0+dsv4.sm80`; the local-version suffix
  identifies this downstream build and does not imply publication on PyPI.

## Acknowledgements

This project is based on [Mini-SGLang](https://github.com/sgl-project/mini-sglang)
and draws implementation guidance from the broader SGLang and vLLM serving
ecosystems.

# Mini-SGLang 0.1.0, DSV4 on SM80

This downstream release serves **DeepSeek V4 Flash only**. Its Python
distribution is `minisgl==0.1.0+dsv4.sm80`; the `+dsv4.sm80` suffix is a
PEP 440 local version intended for source installs, direct wheels, or private
indexes. It is not a claim that this build has been published to PyPI.

The validated platform is one DGX with **8× NVIDIA A100-SXM4-80GB**, tensor
parallelism 8, sm80, CUDA 12.8.2, and NCCL 2.26.2-1. Other hardware and model
families are outside this release contract.

## Install

From the current source checkout:

```bash
python -m pip install -e .
python -c "from importlib.metadata import version; from minisgl.llm import LLM; print(version('minisgl'), LLM.__name__)"
```

Or build and install a wheel:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/minisgl-0.1.0+dsv4.sm80-*.whl
```

WildChat benchmarking has one optional dependency and does not affect the base
CLI:

```bash
python -m pip install -e '.[benchmark]'
```

## Serve and request

The optimized runtime uses release defaults with page size 256, up to 256
running requests, and CUDA graph buckets through M=256. These values were
validated on the DGX A100 platform above; no DSV4 tuning environment variables
or named recipe are required.

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8 --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 1919
```

Query the OpenAI-compatible API from another terminal with the installed
OpenAI client:

```bash
python -c "from openai import OpenAI; c=OpenAI(base_url='http://127.0.0.1:1919/v1', api_key='dummy'); print(c.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'Reply with only 4: 2+2='}], max_tokens=16, temperature=0).choices[0].message.content)"
```

The server provides an OpenAI-compatible, text-only `/v1/chat/completions`
endpoint with streaming and non-streaming responses. It supports string
content, arrays of text content parts, the `system`, `developer`, `user`, and
`assistant` roles, and the sampling fields `max_tokens`,
`max_completion_tokens`, `temperature`, and `top_p`. The minisgl extensions
`top_k` and `ignore_eos` are also supported. When both output-limit spellings
are present, `max_completion_tokens` takes precedence.

For a public service, set an explicit model identity, for example
`--served-model-name deepseek-v4-flash`, and use the ID returned by
`/v1/models`. Without the option, a Hugging Face repo ID remains unchanged and
a local path resolves to its basename (`DeepSeek-V4-Flash` here). Requests may
use only that public ID or the complete configured model path as a compatibility
alias; responses always report the public ID. Authentication is not enabled,
so SDKs may use any non-empty dummy API key.

Custom stop sequences, multiple choices, nonzero presence/frequency
penalties, logprobs, tool/function calling, structured output, multimodal
content, and unknown request options return an OpenAI-style HTTP 400 error;
they are never silently ignored. `stream_options.include_usage` emits an exact
final usage chunk, and non-streaming responses contain the same exact counts.
This endpoint does not claim full OpenAI API parity, and `/metrics` is not
provided.

Interactive shell:

```bash
python -m minisgl.shell --model /models/DeepSeek-V4-Flash --tp-size 8
```

The Python `LLM` example launches its local TP workers from one process:

```bash
python examples/offline_dsv4.py
```

For a slow reference/oracle run, select fallback explicitly before model
construction. Fallback disables the optimized CUDA-graph, PyNCCL, and Marlin
path; it is for correctness diagnosis rather than performance:

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8 --dsv4-runtime fallback
```

## Public benchmarks

The four public benchmarks are intentionally small examples. Edit the constants
near the top of each script to change its workload.

```bash
python benchmark/offline/bench.py
```

The first WildChat shard is cached under `~/.cache/minisgl/benchmarks/`:

```bash
python benchmark/offline/bench_wildchat.py
```

Against the server above, run the simple synthetic benchmark and the
Qwen-format request trace replay. “Qwen-format” describes only the public trace
schema/workload; this release does not serve a Qwen model. Trace files also use
the user benchmark cache by default.

```bash
python benchmark/online/bench_simple.py
python benchmark/online/bench_qwen.py
```

The stable public benchmark API is these four scripts. DSV4 microbenchmarks,
correctness probes, and profiling harnesses live under
[`debug/dsv4/`](debug/dsv4/README.md).

## Runtime contract and recipes

The optimized path includes radix prefix caching, independent SWA cache
lifecycle, chunked prefill, CUDA graph decode replay, Marlin WNA16 MoE, and
PyNCCL TP communication. `fallback` is the explicit oracle. MTP is not included
in this release.

| Recipe | Intent |
| --- | --- |
| `dsv4_sm80_low_m64` | Low active-M or KV-capacity-sensitive serving; graph through M=64. |
| `dsv4_sm80_mid_m128` | Capacity/throughput compromise through M=128. |
| `dsv4_sm80_balanced` | Explicit DGX A100 throughput template through M=256. |
| `dsv4_sm80_long_context_512k` | Low-concurrency 512 Ki-token capability. |
| `dsv4_sm80_1m_smoke` | Single-request 1 Mi-token capability smoke, not a performance recipe. |

Recipes are optional DGX A100 configuration templates, not generic sm80
defaults. Select one on the server with `--dsv4-sm80-recipe NAME`. The server
prints the values supplied by the recipe and lists any fields overridden by
explicit request, graph, or sequence-length arguments.

## Validated baseline

Git tag `v0.0.0` is the immutable pre-cleanup performance baseline. On the
validated DGX A100 platform, its two fresh balanced graph256 runs measured
1.1943/1.1939 requests/s and 1,222.97/1,222.53 output tokens/s for 256 requests
with 1K-token prompts and 1,024-token outputs. See
[`prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md`](prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md)
for the full measurement contract and capacity limits.

The final cleaned release tag is planned as `v0.1.0-dsv4-sm80`; it is created
only after the misc05 final soak and does not exist as part of this misc04 work.

Developer documentation: [`docs/features.md`](docs/features.md) and
[`docs/structures.md`](docs/structures.md).

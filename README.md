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

The optimized runtime and `dsv4_sm80_balanced` recipe are the defaults. The
recipe automatically resolves the validated page size of 256; no DSV4 tuning
environment variables are needed.

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8 --host 0.0.0.0 --port 1919
```

Query the OpenAI-compatible API from another terminal with the installed
OpenAI client:

```bash
python -c "from openai import OpenAI; c=OpenAI(base_url='http://127.0.0.1:1919/v1', api_key='dummy'); print(c.chat.completions.create(model='/models/DeepSeek-V4-Flash', messages=[{'role':'user','content':'Reply with only 4: 2+2='}], max_tokens=16, temperature=0).choices[0].message.content)"
```

Interactive shell:

```bash
python -m minisgl.shell --model /models/DeepSeek-V4-Flash --tp-size 8
```

The Python `LLM` entry also uses one process per TP rank:

```bash
torchrun --standalone --nproc_per_node=8 examples/offline_dsv4.py
```

For a slow reference/oracle run, select fallback explicitly before model
construction. Fallback disables the optimized CUDA-graph, PyNCCL, and Marlin
path; it is for correctness diagnosis rather than performance:

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8 --dsv4-runtime fallback
```

## Public benchmarks

The offline defaults are DSV4 optimized/balanced TP8. Every command accepts
overrides and can write a machine-readable report with `--output`.

```bash
torchrun --standalone --nproc_per_node=8 benchmark/offline/bench.py --request-count 256 --output /tmp/minisgl-offline.json
```

WildChat shards are cached under `~/.cache/minisgl/benchmarks/` unless
`--dataset-cache` or `--dataset-shard` is provided:

```bash
torchrun --standalone --nproc_per_node=8 benchmark/offline/bench_wildchat.py --request-count 32 --output /tmp/minisgl-wildchat.json
```

Against the server above, run the simple synthetic benchmark and the
Qwen-format request trace replay. “Qwen-format” describes only the public trace
schema/workload; this release does not serve a Qwen model. Trace files also use
the user benchmark cache by default.

```bash
python benchmark/online/bench_simple.py --request-count 16 --batch-size 4 --output /tmp/minisgl-online-simple.json
python benchmark/online/bench_qwen.py --request-count 16 --max-concurrency 4 --output /tmp/minisgl-trace-replay.json
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
| `dsv4_sm80_balanced` | Default throughput-oriented configuration through M=256. |
| `dsv4_sm80_long_context_512k` | Low-concurrency 512 Ki-token capability. |
| `dsv4_sm80_1m_smoke` | Single-request 1 Mi-token capability smoke, not a performance recipe. |

Select one with `--dsv4-sm80-recipe NAME` on the server or `--recipe NAME` in
the offline benchmarks.

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

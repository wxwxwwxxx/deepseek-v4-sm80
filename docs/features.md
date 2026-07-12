# DeepSeek V4 release features

This `minisgl==0.1.0+dsv4.sm80` release supports DeepSeek V4 Flash only. The
validated deployment is 8× A100-SXM4-80GB, sm80, tensor parallelism 8, CUDA
12.8.2, and NCCL 2.26.2-1.

## Runtime modes

`optimized` is the default product mode. With no explicit recipe it resolves
`dsv4_sm80_balanced`, including page size 256, radix prefix caching, independent
SWA cache lifecycle, component cache ownership, chunked prefill, CUDA graphs,
Marlin WNA16 routed experts, and PyNCCL communication.

`fallback` is a slow, explicit correctness oracle selected with
`--dsv4-runtime fallback` or `dsv4_runtime_mode="fallback"` before constructing
an `LLM`. It disables the optimized graph, Marlin, PyNCCL, and cache-ownership
path. Internal shape-aware optimized dispatch is not another public runtime
mode.

MTP is not included.

## Serving and shell

The server provides `/v1/models` and the OpenAI-compatible
`/v1/chat/completions` endpoint:

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8
```

The terminal shell uses the same TP8 runtime:

```bash
python -m minisgl.shell --model /models/DeepSeek-V4-Flash --tp-size 8
```

Run `python -m minisgl --help` for the complete CLI.

## Cache and long-context behavior

- Radix prefix caching reuses shared DeepSeek V4 prompt prefixes.
- SWA state has an independent lifecycle from full-context C4/C128/indexer
  state, preserving the model's different attention windows.
- Chunked prefill bounds each extension step; the balanced recipe uses an
  8,192-token chunk budget.
- Public 512 Ki-token and 1 Mi-token recipes deliberately reduce concurrency
  to preserve KV capacity.

## CUDA graph and MoE

The balanced recipe captures decode shapes through active M=256; legal larger
shapes execute eagerly. Marlin WNA16 is the optimized DeepSeek V4 routed-expert
backend on sm80. The fallback oracle retains raw grouped-FP4 weights and does
not depend on Marlin.

## Public recipes

| Recipe | Maximum running requests | Graph max M | Use |
| --- | ---: | ---: | --- |
| `dsv4_sm80_low_m64` | 256 | 64 | Low-M or KV-capacity-sensitive serving. |
| `dsv4_sm80_mid_m128` | 256 | 128 | Capacity/throughput compromise. |
| `dsv4_sm80_balanced` | 256 | 256 | Ordinary throughput-oriented default. |
| `dsv4_sm80_long_context_512k` | 4 | 4 | 512 Ki-token capability. |
| `dsv4_sm80_1m_smoke` | 1 | 1 | 1 Mi-token capability smoke only. |

Explicit request-capacity, graph, sequence, memory, and chunk settings remain
authoritative when supplied. These recipes and their recorded performance are
qualified only on the validated A100 TP8 platform.

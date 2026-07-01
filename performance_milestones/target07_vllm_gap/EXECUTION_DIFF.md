# TARGET 07.1 Execution Diff: mini vs vLLM

Scope: DeepSeek V4 Flash, A100/sm80, TP8, page/block size 256, batch size 4,
4096 prompt tokens. This document compares execution structure only. It does
not make vLLM a mini-sglang runtime dependency.

## High-Level Diff

| Area | mini current path | vLLM observed path | Decision |
| --- | --- | --- | --- |
| Benchmark fairness | mini evidence used warmup=0; DSV4 graph disabled | vLLM evidence used warmup=1, chunked prefill, graph capture sizes 1/2/4 | adapt |
| MoE runner boundary | Python model calls gate, grouped routed experts, shared experts, optional late reduce | `FusedMoE` runner is behind `torch.ops.vllm.moe_forward*` custom ops | adapt |
| MoE reduction | v1_moe can reduce routed+shared once when enabled | runner late-reduces combined output unless fused kernel already reduced | port |
| Shared experts overlap | runs inline with routed expert path | optional auxiliary CUDA stream in shared expert runner | adapt |
| MoE quantization | bf16-direct exact path around fp4 weights, no activation quantization by default | MXFP4 methods select FP8/FP4/Marlin/FlashInfer/DeepGEMM backends | defer |
| CUDA graph | simple fixed-batch graph runner exists but DSV4 forcibly disables it | runtime dispatcher selects FULL/PIECEWISE/NONE with valid batch descriptors | adapt |
| Communication | benchmark uses torch/NCCL path by default; PyNCCL exists but is not default | custom-op collectives, PyNCCL, custom all-reduce, capture-aware buffers | adapt |
| DSV4 attention wrapper | model directly builds metadata and calls backend/fallback kernels | attention has custom op boundary plus graph-visible q/kv norm and projection | adapt |
| SM80 sparse decode | mini has bf16 exact sparse attention kernels/fallbacks | vLLM reference decode uses split-K sparse attention after gather/dequant | adapt |
| SM80 sparse prefill | mini uses exact bf16 path, still expensive | vLLM reference prefill can materialize large gathered KV | reject |
| Precision policy | exact bf16-direct first, preserve fp32, TF32 only explicit experiment | vLLM uses FP8 KV/activation paths in places | defer |

## vLLM Design Points

### FusedMoE

vLLM wires DeepSeek V4 MoE through `FusedMoE` in
`/workspace/vllm-dsv4-docker/vllm/model_executor/models/deepseek_v4.py`.
The key handoff passes shared experts, gate, top-k, scoring function, correction
bias, hash routing table, swiglu limit, and `router_logits_dtype=torch.float32`
into a single runner-backed module. The forward path then calls either the
internal-router `FusedMoE` or computes router logits and passes them to it.

mini currently does the same logical stages in
`python/minisgl/models/deepseek_v4.py`: `DSV4MoEGate`,
`DSV4FusedRoutedExperts`, `DSV4SharedExperts`, then optional single all-reduce
after routed+shared in `DSV4MoE.forward`.

Decision: `adapt`.

Action: keep mini's exact bf16-direct semantics, but introduce a smaller
runner boundary with explicit prepare/experts/finalize phases and reusable
workspace. Do not copy vLLM's full quantization framework. The immediate TARGET
07.2 value is reducing Python and collective overhead around the existing exact
path; TARGET 07.3 can then replace the grouped expert kernel behind the same
runner boundary.

### Late MoE All-Reduce

vLLM's `MoERunner` has two mutually exclusive all-reduce points: if the fused
kernel already reduced routed output, it reduces shared output separately;
otherwise it adds routed+shared locally and late-reduces the sum.

mini v1_moe already has the same important shape: when
`MINISGL_DSV4_SM80_V1_MOE` is active, routed and shared experts skip their own
reduce and `DSV4MoE.forward` reduces once after the add.

Decision: `port`.

Action: preserve this late-reduce invariant as a named contract and add profiling
labels for every all-reduce site. The next implementation should make it hard to
accidentally reintroduce separate routed and shared reductions.

### Shared Expert Overlap

vLLM can run shared experts in an auxiliary CUDA stream when token count,
quantization method, and parallel config allow it. The runner explicitly applies
shared experts in no-overlap, kernel-internal-overlap, or multi-stream-overlap
positions.

mini shared experts run inline today.

Decision: `adapt`.

Action: defer the actual stream overlap until after fair profiling, but shape the
future MoE runner so shared expert launch order is explicit. This is lower
priority than communication and graph replay because the current profile is
dominated by collectives, grouped FP4 MoE, and small PyTorch kernels.

### MXFP4, FP8, DeepGEMM, Marlin Backends

vLLM's DeepSeek V4 path uses MXFP4 MoE methods and backend selection. Some
paths use FP8 activation/KV behavior or non-exact dequantization choices. vLLM
also has a DeepGEMM MegaMoE path, but the DeepSeek V4 model code forces the
standard `FusedMoE` path for SM80/reference kernels.

mini's first phase must keep bf16-direct exact by default and must not silently
lower existing fp32 computation. TF32 is an explicit experiment only.

Decision: `defer` for quantized activation/backend policy, `reject` for SM80
default DeepGEMM MegaMoE.

Action: do not port vLLM's activation quantization as part of TARGET 07.2. Use
vLLM only as a design reference for separating routing, dispatch, expert compute,
and combine. Precision lanes belong to TARGET 07.4.

### CUDA Graph Dispatcher

mini has a simple decode graph runner in `python/minisgl/engine/graph.py`, but
`python/minisgl/engine/engine.py` currently forces DeepSeek V4
`cuda_graph_bs=[]` and `cuda_graph_max_bs=0`. So DSV4 always takes eager decode
today.

vLLM has a runtime `CudagraphDispatcher` that computes valid batch descriptors,
pads to capture sizes, and dispatches FULL, PIECEWISE, or NONE through forward
context. The fair vLLM profile shows CUDA graph events, while the mini profile
does not.

Decision: `adapt`.

Action: TARGET 07.2 should not blindly enable mini's existing graph runner for
DSV4. First make DSV4 decode metadata capture-stable, then allow graph sizes
`1,2,4` for uniform decode batch descriptors. A small dispatcher is enough for
mini; LoRA/speculative/piecewise compile machinery can be omitted until needed.

### Communication Custom Ops and Custom All-Reduce

mini's current benchmark config uses `use_pynccl=false`, so TP collectives go
through `TorchDistributedImpl` and in-place `dist.all_reduce`. PyNCCL exists,
but is not the measured default path.

vLLM wraps collectives as custom ops where supported, uses a device
communicator, tries custom all-reduce/FlashInfer/symmetric memory/PyNCCL before
falling back to torch distributed, and registers custom all-reduce buffers during
CUDA graph capture. vLLM custom all-reduce is single-node focused and requires
supported world sizes, P2P/NVLink connectivity for larger groups, weak
contiguity, and input byte sizes divisible by 16.

Decision: `adapt`.

Action: TARGET 07.2 should first label and count every mini collective, then run
an exact-path PyNCCL opt-in comparison. If PyNCCL is correct and faster, make it
the benchmark path or a DSV4 TP8 default. A vLLM-style custom op boundary is
worth adapting for CUDA graph capture, but the C++ custom all-reduce kernel
itself should be ported only after PyNCCL and graph replay are measured.

### DSV4 Attention Wrapper

mini builds DSV4 attention metadata in Python/Torch, stores SWA/compressed/indexer
KV, and calls the DSV4 backend from `DSV4Attention.forward`. It keeps bf16 exact
KV cache semantics.

vLLM lifts q/kv RMSNorm and `wq_b` out of the opaque attention custom op so the
surrounding graph can still see those ops, then calls
`torch.ops.vllm.deepseek_v4_attention` for KV insert, indexer/compressor overlap,
and MLA attention. For SM80/reference kernels, it uses bf16 inverse RoPE and
attention reference paths rather than FlashMLA FP8.

Decision: `adapt`.

Action: keep mini's bf16 KV cache by default. Borrow the structural idea:
separate graph-visible projections/norms from an opaque attention op boundary,
and make metadata replay stable. Do not import vLLM attention as a dependency.

### SM80 Sparse Decode

vLLM's SM80 reference decode path uses a split-K sparse attention kernel after
gather/dequant. This improves decode parallelism for low batch/small token
counts and is relevant to mini's decode-heavy bottleneck.

mini already has DSV4 sparse attention hooks and bf16 exact kernels/fallbacks.

Decision: `adapt`.

Action: compare kernel launch shape and memory layout after the fair nsys pass.
If attention remains high after communication/graph work, adapt the split-K idea
to mini's bf16 cache layout without adopting FP8 cache semantics.

### SM80 Sparse Prefill

vLLM's SM80/reference prefill path can materialize large gathered KV tensors.
This path has already been observed as OOM-prone for the current work.

Decision: `reject`.

Action: do not port the vLLM reference sparse prefill path as mini default. If
prefill still needs work, design a memory-bounded bf16-direct path instead.

### Chunked Prefill

vLLM fair profile used chunked prefill with `max_num_batched_tokens=4096`.
mini existing evidence did not isolate this setting. TARGET 07.1 scripts run
mini nsys both with the default prefill path and with `MAX_EXTEND_TOKENS=4096`.

Decision: `adapt`.

Action: treat chunked prefill as a fairness and scheduler-shape variable first.
Only after the fair pair is available should mini change default scheduling.

## Next Work Gate

Enter TARGET 07.2 only after recording:

1. mini 4096/1024/bs4 warmup=1 macro result
2. vLLM 4096/1024/bs4 warmup=1 macro result
3. mini 4096/128/bs4 warmup=1 nsys, default prefill
4. mini 4096/128/bs4 warmup=1 nsys, `MAX_EXTEND_TOKENS=4096`
5. vLLM 4096/128/bs4 warmup=1 nsys

Expected TARGET 07.2 first implementation axis: communication labeling and
PyNCCL/custom-op graph-readiness, followed by DSV4 decode CUDA graph enablement
for capture sizes `1,2,4`. MoE V2 should wait until the fair communication and
graph deltas are known.

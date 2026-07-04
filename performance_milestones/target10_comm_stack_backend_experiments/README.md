# TARGET 10.2: DSV4 SM80 Communication Stack Backend Experiments

Status: complete for the current backend gate. No low-precision route, attention
kernel change, or prefix-cache ownership change was made.

Decision: keep backend changes opt-in. Do not promote a new communication
backend into the fixed DeepSeek V4 Flash A100/sm80 baseline yet.

The best candidate in this pass is the existing PyNCCL opt-in with a size
threshold:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

This keeps the TARGET 10.15 BF16 MoE reduce boundary fixed, sends 128 MiB
hidden all-reduces through PyNCCL direct, and sends about 20 MiB serving-wave
hidden all-reduces through the mini PyNCCL symmetric-buffer path. It passed
text smoke and all four full-model macro scenarios with zero eager decode
fallback, but the macro evidence is a single run and the gains are modest. It
needs repeat-stable validation and owner timing before promotion.

## Fixed Baseline

All experiments used the fixed TARGET 10.15 path:

- variant: `dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16`
- explicit env: `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`
- page size: 256
- num pages: 128
- radix prefix cache: enabled
- component loc ownership: enabled
- CUDA graph buckets: `1 2 4 8 16`

Raw artifacts:

- microbench, trace replay, partial probe:
  `raw/tp8_comm_backend_probe.json`
- PyNCCL threshold text smoke:
  `raw/text_smoke_pynccl_threshold32m.json`
- current-run Torch/NCCL fixed baseline macro:
  `raw/macro_torch_nccl_fixed_baseline/summary.json`
- current-run PyNCCL threshold32m macro:
  `raw/macro_pynccl_threshold32m/summary.json`
- probe script:
  `scripts/tp8_comm_backend_probe.py`

## Backend Inventory

### Mini TorchDistributedImpl

`python/minisgl/distributed/impl.py`:

- `all_reduce(x)` is in-place `torch.distributed.all_reduce(..., SUM)` and
  returns `x`.
- `all_gather(x)` allocates a new output tensor with dim0 multiplied by TP size
  and calls `torch.distributed.all_gather_into_tensor`.
- Communication stats are recorded above the backend by label, op, dtype,
  input shape, output shape, count, and logical bytes.

### Mini PyNCCLDistributedImpl

`python/minisgl/distributed/impl.py` and `python/minisgl/kernel/pynccl.py`:

- `all_reduce(x)` is in-place through `comm.all_reduce(x, "sum")`.
- `all_gather(x)` allocates a new result tensor in Python, then calls
  `comm.all_gather(result, x)`.
- `init_pynccl(..., max_size_bytes)` clamps the requested symmetric workspace to
  `MINISGL_PYNCCL_MAX_BUFFER_SIZE` / `ENV.PYNCCL_MAX_BUFFER_SIZE`; default is
  1 GiB.
- Engine integration uses PyNCCL when `config.use_pynccl` is true. The benchmark
  fixed baseline uses Torch/NCCL unless `--use-pynccl` is passed.

### pynccl.cu

`python/minisgl/kernel/csrc/src/pynccl.cu`:

- Supported NCCL dtypes: fp16, BF16, fp32.
- Constructor initializes an NCCL communicator, allocates `m_sym_mem` with
  `ncclMemAlloc(max_bytes)`, and registers it with
  `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`.
- `all_reduce` behavior:
  - if `size_bytes <= m_max_bytes`, use the internal symmetric-memory buffer;
  - if input pointer differs from the internal buffer, copy input to buffer;
  - run `ncclAllReduce` in-place on the buffer;
  - copy the reduced buffer back to the input tensor;
  - otherwise, when `size_bytes > m_max_bytes`, call direct in-place
    `ncclAllReduce` on the input tensor.
- `all_gather` behavior:
  - never uses the internal buffer;
  - calls `ncclAllGather(src_ptr, dst_ptr, ...)` directly into the output tensor.
- Mini PyNCCL currently does not expose reduce-scatter, send/recv, or a custom
  all-reduce path.

### vLLM Reference Ideas

Source inspected under `/workspace/vllm-dsv4-docker/vllm/distributed/`.

| vLLM idea | Current applicability to mini DSV4 sm80 owner shapes |
|---|---|
| PyNcclCommunicator | Complete NCCL wrapper. It supports all-reduce, all-gather, all-gatherv, reduce-scatter, reduce-scatterv, send/recv, broadcast, and NCCL comm windows. Mini has only all-reduce/all-gather today. |
| custom all-reduce | Complete out-of-place all-reduce using CUDA IPC/P2P shared buffers, graph buffer registration, and P2P checks. It supports TP8, but the default max size is 8 MiB when sm80 is not in the symmetric-memory max-size table. The current hot hidden shapes are about 20 MiB and 128 MiB, so it does not naturally cover them without a separate port/tuning target. |
| quick all-reduce | Not applicable on CUDA/A100. vLLM quick all-reduce is ROCm MI300-oriented and `quick_ar` is disabled for CUDA. |
| torch symmetric memory | vLLM `SymmMemCommunicator` supports selected capabilities 9.0/10.0/10.3, not sm80. Not a direct A100 route. |
| CUDA IPC/P2P | Useful building block. All local peer-access pairs reported accessible in this container, but raw peer copies are not a collective and were not treated as a drop-in all-reduce/all-gather backend. |
| reduce-scatter | vLLM exposes it, mini does not. TARGET 10.1 did not find a required reduce-scatter boundary change for the current DSV4 TP path, so this target did not alter logical boundaries. |

## Pure Communication Microbench

Command:

```bash
timeout 1800 torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target10_comm_stack_backend_experiments/scripts/tp8_comm_backend_probe.py \
  --output performance_milestones/target10_comm_stack_backend_experiments/raw/tp8_comm_backend_probe.json \
  --warmup 5 --iterations 30 --trace-iterations 3 --graph-replays 16
```

Environment: 8x NVIDIA A100-SXM4-80GB, sm80, Torch 2.9.1+cu128. BF16/fp32
correctness was checked against Torch/NCCL for every row. All rows had
`max_abs=0`.

| Owner set | Shape / dtype / op | Backend | Median us | P95 us | Achieved GB/s | Symm D2D copy per call |
|---|---|---:|---:|---:|---:|---:|
| attention `wo_b`, MoE reduce-once BF16, embedding, historical | `[16384,4096]` BF16 all-reduce | Torch/NCCL | 1351.01 | 1587.07 | 99.3 | 0 |
| same | same | PyNCCL direct | 1179.94 | 1321.70 | 113.8 | 0 |
| same | same | PyNCCL symmetric | 1450.91 | 1980.93 | 92.5 | 256 MiB, est. 326.75 us |
| attention `wo_b`, MoE reduce-once BF16, embedding, serving wave | `[2496,4096]` BF16 all-reduce | Torch/NCCL | 323.63 | 342.91 | 63.2 | 0 |
| same | same | PyNCCL direct | 256.96 | 261.89 | 79.6 | 0 |
| same | same | PyNCCL symmetric | 239.22 | 258.91 | 85.5 | 39 MiB, est. 65.28 us |
| lm_head fallback, historical | `[4,16160] -> [32,16160]` fp32 all-gather | Torch/NCCL | 64.64 | 76.80 | 32.0 | 0 |
| same | same | PyNCCL direct | 44.34 | 51.52 | 46.7 | 0 |
| same | same | PyNCCL symmetric-config | 60.64 | 69.06 | 34.1 | 0, direct output |
| lm_head fallback, serving wave | `[16,16160] -> [128,16160]` fp32 all-gather | Torch/NCCL | 111.68 | 117.22 | 74.1 | 0 |
| same | same | PyNCCL direct | 85.10 | 115.01 | 97.2 | 0 |
| same | same | PyNCCL symmetric-config | 83.09 | 86.27 | 99.6 | 0, direct output |

Interpretation:

- PyNCCL direct is faster than Torch/NCCL in isolated single-collective
  measurements, but this did not translate into a strong no-weight historical
  trace win.
- Full symmetric-buffer all-reduce is bad for the 128 MiB hidden shape because
  copy-in/copy-out adds about 327 us per call.
- Symmetric-buffer all-reduce is competitive for the about 20 MiB serving-wave
  shape, even after the two D2D copies.
- PyNCCL all-gather writes directly to the output; the symmetric-buffer setting
  does not add D2D copies for all-gather.

## D2D Copy Overhead Accounting

Standalone D2D copy microbench:

| Shape / dtype | One D2D copy median us | One-copy bandwidth | Symm all-reduce copy bytes per call | Two-copy estimate |
|---|---:|---:|---:|---:|
| `[16384,4096]` BF16, 128 MiB | 163.38 | 821.5 GB/s | 256 MiB | 326.75 us |
| `[2496,4096]` BF16, 19.5 MiB | 32.64 | 626.4 GB/s | 39 MiB | 65.28 us |

Trace-level copy amplification:

| Route | Historical trace D2D copies | Serving trace D2D copies | Finding |
|---|---:|---:|---|
| PyNCCL symmetric for all hidden all-reduces | 348.0 GiB | 185.6 GiB | Historical copy overhead dominates and should be rejected. |
| Threshold32m candidate | 0 GiB historical, because 128 MiB is direct | 185.6 GiB serving, because 19.5 MiB fits | This is the only tested PyNCCL route worth keeping as opt-in. |
| Prefix macro threshold32m estimate | about 65.2 GiB | N/A | Only `[1024,4096]` BF16 hidden all-reduces fit the 32 MiB window; `[9216,4096]` stays direct. |

## No-Weight Trace Replay

The synthetic trace replays the owner order:

```text
embedding all-reduce
43 * (attention wo_b all-reduce, MoE reduce-once all-reduce)
lm_head all-gather
```

Then it repeats the body to match the owner counts:

- `historical_4096_128_bs4`: 16 body repeats
- `serving_mixed_112req_wave16`: 56 body repeats

| Scenario | Backend | Eager median ms | Graph replay median ms | Graph capture | Symm D2D copies |
|---|---|---:|---:|---|---:|
| historical_4096_128_bs4 | Torch/NCCL | 1629.37 | 1664.03 | ok | 0 |
| historical_4096_128_bs4 | PyNCCL direct | 1626.37 | 1664.41 | ok | 0 |
| historical_4096_128_bs4 | PyNCCL symmetric | 2004.99 | 2000.10 | ok | 348.0 GiB |
| serving_mixed_112req_wave16 | Torch/NCCL | 1224.80 | 1345.59 | ok | 0 |
| serving_mixed_112req_wave16 | PyNCCL direct | 1227.10 | 1356.51 | ok | 0 |
| serving_mixed_112req_wave16 | PyNCCL symmetric | 1122.49 | 1107.69 | ok | 185.6 GiB |

Gate result:

- reject all-symmetric routing for current owner shapes;
- do not promote direct-only routing because no-weight trace is neutral;
- keep thresholded PyNCCL as the only full-model candidate: direct for 128 MiB,
  symmetric for about 20 MiB and smaller.

## Partial Runtime Probe

The probe used the real `DistributedCommunicator` and `PyNCCLDistributedImpl`
plugin APIs with synthetic tensors, labels, stats, and CUDA graph capture.

| Backend | BF16 all-reduce correct vs Torch/NCCL | fp32 all-gather correct vs Torch/NCCL | Graph capture | Stats labels |
|---|---|---|---|---|
| Torch/NCCL | yes | yes | yes | yes |
| PyNCCL direct | yes | yes | yes | yes |
| PyNCCL symmetric | yes | yes | yes | yes |

The partial probe caught no ABI, stream ordering, lifecycle, dtype, or graph
capture blocker.

## Graph-Capture Compatibility

| Layer | Torch/NCCL | PyNCCL direct | PyNCCL symmetric |
|---|---|---|---|
| Pure microbench single collective | ok | ok | ok |
| No-weight trace replay | ok | ok | ok |
| DistributedCommunicator partial probe | ok | ok | ok |
| Full model candidate `threshold32m` | N/A | ok | ok |

Full-model graph replay stayed zero-eager for the candidate:

| Scenario | Candidate graph replay/eager |
|---|---:|
| historical_4096_128_bs4 | 127 / 0 |
| historical_4096_1024_bs4 | 1023 / 0 |
| serving_mixed_112req_wave16 | 441 / 0 |
| prefix_multi_112req_wave16 | 49 / 0 |

## BF16/fp32 Correctness

| Gate | Result |
|---|---|
| Pure BF16 all-reduce correctness vs Torch/NCCL | pass, max_abs 0 |
| Pure fp32 all-gather correctness vs Torch/NCCL | pass, max_abs 0 |
| DistributedCommunicator BF16/fp32 probe | pass |
| Full model PyNCCL threshold32m text smoke | pass |

Text smoke command:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M \
timeout 1200 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --output performance_milestones/target10_comm_stack_backend_experiments/raw/text_smoke_pynccl_threshold32m.json \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 32 --fail-on-warning --use-pynccl
```

Result: pass. The three prompts produced sane Chinese/math, English sky-color,
and Hangzhou outputs. Smoke graph replay/eager was 9 / 0.

## CUDA P2P/IPC Feasibility

The probe checked `torch.cuda.can_device_access_peer` for all 8 visible A100s:
all local pairs are peer-accessible.

This is useful for a future custom communicator, but raw P2P or IPC peer-copy
was not treated as a complete collective. A promotable path would still need:

- IPC handle exchange;
- full all-reduce/all-gather semantics;
- rank synchronization;
- stream ordering;
- graph registration or graph-safe fixed addresses;
- correctness against Torch/NCCL.

vLLM custom all-reduce already solves much of that structure, but its current
source thresholds do not naturally cover the current hot hidden shapes on sm80.

## Full Model Macro A/B

Baseline command: Torch/NCCL fixed BF16 path, output
`raw/macro_torch_nccl_fixed_baseline/`.

Candidate command: PyNCCL threshold32m fixed BF16 path, output
`raw/macro_pynccl_threshold32m/`.

Candidate env:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
--use-pynccl
```

Both runs used one repeat, zero warmup repeats, page size 256, num pages 128,
radix prefix cache, component loc ownership, and CUDA graph buckets
`1 2 4 8 16`.

| Scenario | Torch elapsed s | Candidate elapsed s | Elapsed delta | Torch decode tok/s | Candidate decode tok/s | Decode delta | Torch E2E tok/s | Candidate E2E tok/s | E2E delta | Graph candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_4096_128_bs4 | 9.723 | 9.546 | -1.82% | 182.31 | 189.82 | +4.12% | 52.66 | 53.64 | +1.85% | 127/0 |
| historical_4096_1024_bs4 | 29.824 | 28.757 | -3.58% | 183.04 | 192.33 | +5.07% | 137.34 | 142.43 | +3.71% | 1023/0 |
| serving_mixed_112req_wave16 | 15.964 | 15.241 | -4.52% | 285.01 | 301.86 | +5.91% | 175.40 | 183.71 | +4.74% | 441/0 |
| prefix_multi_112req_wave16 | 6.701 | 6.617 | -1.26% | 647.15 | 677.07 | +4.62% | 133.72 | 135.42 | +1.27% | 49/0 |

Communication stats were unchanged, as expected, because this target changes
backend only, not owner boundaries or dtype:

| Scenario | Candidate comm count | Candidate comm GiB | Main dtype state |
|---|---:|---:|---|
| historical_4096_128_bs4 | 704 | 87.015 | attention BF16, MoE BF16, embedding BF16, lm_head fp32 |
| historical_4096_1024_bs4 | 704 | 87.015 | same |
| serving_mixed_112req_wave16 | 4928 | 93.209 | same |
| prefix_multi_112req_wave16 | 4928 | 81.994 | same |

Owner timing/profile was not run for the candidate because this pass does not
promote it. The macro reports include communication stats, bucket coverage, and
graph replay/eager state.

## Decisions

| Route | Decision | Reason |
|---|---|---|
| Torch/NCCL fixed BF16 baseline | keep default | Correct, graph-safe, stable baseline. |
| PyNCCL direct only | keep opt-in / do not promote | Isolated microbench wins, but no-weight trace is neutral. |
| PyNCCL symmetric for all hidden all-reduces | reject | 128 MiB hidden trace pays 348 GiB D2D copies and is about 23% slower in no-weight replay. |
| PyNCCL threshold32m | keep opt-in | Passes smoke and full macro, with single-run E2E +1.27% to +4.74% vs same-run Torch baseline. Not repeat-stable yet, and gains are not large enough to promote blindly. |
| vLLM custom all-reduce direct port | defer | Complete backend idea, but current sm80 hot shapes exceed the natural default threshold. Needs a separate port/threshold study. |
| vLLM quick all-reduce | reject for A100/sm80 | ROCm MI300 route, not CUDA/A100. |
| CUDA P2P/IPC raw peer copy | reject as standalone backend | Peer access is available, but raw P2P/IPC lacks collective semantics and synchronization. |
| reduce-scatter boundary change | reject for this target | TARGET 10.1 did not identify a required logical boundary difference. |

## Next Steps

1. Keep `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1` as the fixed dtype path for any
   follow-up backend work.
2. If backend tuning continues, run a repeat-stable gate for
   `--use-pynccl + MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M`, at least two repeats of
   `historical_4096_1024_bs4` and `serving_mixed_112req_wave16`, plus one prefix
   health run.
3. Add owner timing/profile only if the repeat gate stays positive; current
   evidence is enough to keep opt-in but not enough to promote.
4. Consider a true per-owner/per-size routing layer if PyNCCL remains useful:
   Torch/NCCL for large prefill-style hidden all-reduces, PyNCCL symmetric only
   for smaller fixed-shape decode owners, and direct-output PyNCCL all-gather
   only if allocator and graph behavior remain clean.
5. Treat a vLLM custom all-reduce port as a separate target. It should start
   from size coverage and graph registration, not from raw P2P copies.

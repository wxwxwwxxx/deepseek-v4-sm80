# TARGET 10.25: DSV4 SM80 Communication Size/Owner Routing

Status: complete for the current gate.

Decision: keep `MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M --use-pynccl` as the
documented opt-in candidate. Do not add a production per-owner/per-size routing
hook in this target. The existing global PyNCCL threshold repeated positively
in full model gate runs, but explicit owner/size policies did not beat the
global threshold in no-weight replay.

No low-precision path, attention kernel, prefix/SWA ownership, or custom P2P
collective was changed.

## Artifacts

- Threshold32m Torch/NCCL gate:
  `raw/gate_torch_nccl_hist_serving_r2/`,
  `raw/gate_torch_nccl_prefix_r1/`
- Threshold32m PyNCCL gate:
  `raw/gate_pynccl_threshold32m_hist_serving_r2/`,
  `raw/gate_pynccl_threshold32m_prefix_r1/`
- Text smoke:
  `raw/text_smoke_pynccl_threshold32m.json`
- Route microbench/no-weight replay:
  `raw/tp8_comm_route_policy_probe.json`
- Probe script extended:
  `../target10_comm_stack_backend_experiments/scripts/tp8_comm_backend_probe.py`

## Threshold32m Repeat-Stable Gate

Fixed path:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
--variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Candidate:

```bash
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M --use-pynccl
```

| Scenario | Backend | Repeats | Elapsed s | E2E out tok/s | Decode tok/s | Graph replay/eager | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `historical_4096_1024_bs4` | Torch/NCCL | 2 | 61.492 | 133.220 | 182.340 | 2046 / 0 | baseline |
| `historical_4096_1024_bs4` | PyNCCL threshold32m | 2 | 59.136 | 138.528 | 192.135 | 2046 / 0 | +3.98% E2E, +5.37% decode |
| `serving_mixed_112req_wave16` | Torch/NCCL | 2 | 31.712 | 176.591 | 284.701 | 882 / 0 | baseline |
| `serving_mixed_112req_wave16` | PyNCCL threshold32m | 2 | 30.818 | 181.715 | 302.457 | 882 / 0 | +2.90% E2E, +6.24% decode |
| `prefix_multi_112req_wave16` | Torch/NCCL | 1 | 8.109 | 110.494 | 409.046 | 49 / 0 | health pass, hit rate 0.857 |
| `prefix_multi_112req_wave16` | PyNCCL threshold32m | 1 | 8.013 | 111.819 | 419.753 | 49 / 0 | health pass, hit rate 0.857 |

Per-repeat elapsed:

| Scenario | Repeat | Torch/NCCL s | Threshold32m s | Delta |
|---|---:|---:|---:|---:|
| `historical_4096_1024_bs4` | 0 | 31.035 | 29.821 | -3.91% |
| `historical_4096_1024_bs4` | 1 | 29.841 | 28.710 | -3.79% |
| `serving_mixed_112req_wave16` | 0 | 15.792 | 15.500 | -1.85% |
| `serving_mixed_112req_wave16` | 1 | 15.919 | 15.317 | -3.78% |

Gate result: threshold32m is repeat-positive for the required repeated
scenarios and remains zero-eager. This is enough to test route policies, but not
enough to promote into the default bundle because the route-specific cheap gate
below did not find a better explicit route and owner timing was not re-run.

## Existing PyNCCL Interface Reuse

Mini already has the mechanism needed for the threshold candidate:

- `PyNCCLDistributedImpl` wraps the existing `init_pynccl(...)` communicator.
- `MINISGL_PYNCCL_MAX_BUFFER_SIZE` is parsed through `ENV.PYNCCL_MAX_BUFFER_SIZE`.
- `pynccl.cu` `NCCLWrapper::all_reduce` routes by input size:
  `size_bytes <= m_max_bytes` uses the internal symmetric buffer with D2D
  copy-in/copy-out, while larger tensors use direct in-place `ncclAllReduce`.
- `pynccl.cu` `all_gather` writes directly to the output tensor and does not
  use the symmetric buffer.
- `DistributedCommunicator` already records `label/op/dtype/shape/bytes`, so it
  is the right layer for future owner/shape routing.

No C++ communication stack rewrite was needed. The probe only added synthetic
route policies around the existing Torch/NCCL and PyNCCL implementations.

## Route Policy Design

Policies tested:

| Policy | Decision rule |
|---|---|
| `torch_all` | Torch/NCCL for all collectives. |
| `pynccl_threshold32m` | Existing global PyNCCL path with `m_max_bytes=32M`; small all-reduce uses symmetric buffer, large all-reduce uses direct NCCL, all-gather writes direct output. |
| `route_small_hidden_to_pynccl` | PyNCCL threshold32m only for BF16 hidden all-reduce owners with input bytes `<=32M`; Torch/NCCL otherwise. |
| `route_hidden_to_pynccl` | PyNCCL threshold32m for all selected BF16 hidden all-reduce owners; Torch/NCCL for all-gather. |
| `route_gather_to_pynccl` | PyNCCL threshold32m only for `dsv4.lm_head_all_gather`; Torch/NCCL otherwise. |

Owner scope:

- `dsv4.attn.wo_b.row_parallel_projection_all_reduce`
- `dsv4.v1_moe_reduce_once_all_reduce`
- `dsv4.embedding_all_reduce`
- `dsv4.lm_head_all_gather`

The route decisions use only op, label, dtype, and input bytes.

## Microbench

Selected pure communication rows from
`raw/tp8_comm_route_policy_probe.json`, TP8 A100, 30 iterations:

| Owner/shape | Backend | Median us | P95 us | D2D copy/call | Correct | Graph |
|---|---:|---:|---:|---:|---:|---:|
| hidden BF16 `[16384,4096]` all-reduce | Torch/NCCL | 1328.10 | 1628.80 | 0 | yes | yes |
| hidden BF16 `[16384,4096]` all-reduce | PyNCCL threshold32m direct | 1186.19 | 3824.54 | 0 | yes | yes |
| hidden BF16 `[2496,4096]` all-reduce | Torch/NCCL | 326.86 | 422.62 | 0 | yes | yes |
| hidden BF16 `[2496,4096]` all-reduce | PyNCCL threshold32m symmetric | 246.64 | 256.90 | 39.0 MiB | yes | yes |
| hidden BF16 `[1024,4096]` all-reduce | Torch/NCCL | 177.92 | 221.44 | 0 | yes | yes |
| hidden BF16 `[1024,4096]` all-reduce | PyNCCL threshold32m symmetric | 125.94 | 156.32 | 16.0 MiB | yes | yes |
| hidden BF16 `[9216,4096]` all-reduce | Torch/NCCL | 739.39 | 753.86 | 0 | yes | yes |
| hidden BF16 `[9216,4096]` all-reduce | PyNCCL threshold32m direct | 652.85 | 681.98 | 0 | yes | yes |
| fp32 `[16,16160] -> [128,16160]` all-gather | Torch/NCCL | 108.61 | 120.67 | 0 | yes | yes |
| fp32 `[16,16160] -> [128,16160]` all-gather | PyNCCL direct output | 85.01 | 104.19 | 0 | yes | yes |

Interpretation: isolated collectives still favor PyNCCL median latency, but the
large `[16384,4096]` row has a tail outlier in this run and no-weight trace is
the decisive cheap gate.

## No-Weight Replay

The route replay issues the owner order:

```text
embedding all-reduce
43 * (attention wo_b all-reduce, MoE reduce-once all-reduce)
lm_head all-gather
```

It covers historical, serving, and a prefix-like mixed-shape trace:

- historical traces: `[16384,4096]`, batch 4, 16 forward bodies;
- serving trace: `[2496,4096]`, batch 16, 56 forward bodies;
- prefix trace: 8 bodies at `[9216,4096]` and 48 bodies at `[1024,4096]`.

Graph replay median:

| Scenario | `torch_all` ms | `pynccl_threshold32m` ms | `route_small_hidden_to_pynccl` ms | `route_hidden_to_pynccl` ms | `route_gather_to_pynccl` ms |
|---|---:|---:|---:|---:|---:|
| `historical_4096_128_bs4` | 1659.00 | 1672.69 | 1658.67 | 1670.04 | 1657.25 |
| `historical_4096_1024_bs4` | 1661.24 | 1672.46 | 1659.52 | 1671.74 | 1657.50 |
| `serving_mixed_112req_wave16` | 1347.43 | 1165.34 | 1215.28 | 1214.35 | 1339.70 |
| `prefix_multi_112req_wave16` | 1100.10 | 916.86 | 950.75 | 950.85 | 1106.02 |

D2D copy totals in no-weight replay:

| Scenario | Threshold32m / small-hidden / hidden route | Torch / gather-only route |
|---|---:|---:|
| historical traces | 0 GiB | 0 GiB |
| `serving_mixed_112req_wave16` | 185.55 GiB | 0 GiB |
| `prefix_multi_112req_wave16` | 65.25 GiB | 0 GiB |

No explicit route beat `pynccl_threshold32m` on the serving and prefix traces.
`route_small_hidden_to_pynccl` avoids threshold32m's historical replay loss, but
it gives up the all-gather and larger direct-PyNCCL benefits that make the
global threshold fastest on serving/prefix no-weight replay. Therefore no
production owner/size route survived the cheap gate.

## Route Stats By Owner/Shape

Selected backend summary:

| Owner/shape | `pynccl_threshold32m` | `route_small_hidden_to_pynccl` | `route_hidden_to_pynccl` | `route_gather_to_pynccl` |
|---|---|---|---|---|
| hidden BF16 `[16384,4096]` all-reduce | PyNCCL direct | Torch/NCCL | PyNCCL direct | Torch/NCCL |
| hidden BF16 `[2496,4096]` all-reduce | PyNCCL symmetric | PyNCCL symmetric | PyNCCL symmetric | Torch/NCCL |
| hidden BF16 `[1024,4096]` all-reduce | PyNCCL symmetric | PyNCCL symmetric | PyNCCL symmetric | Torch/NCCL |
| hidden BF16 `[9216,4096]` all-reduce | PyNCCL direct | Torch/NCCL | PyNCCL direct | Torch/NCCL |
| `lm_head` fp32 `[4,16160]` all-gather | PyNCCL direct output | Torch/NCCL | Torch/NCCL | PyNCCL direct output |
| `lm_head` fp32 `[16,16160]` all-gather | PyNCCL direct output | Torch/NCCL | Torch/NCCL | PyNCCL direct output |

Full-model gate communication counters stayed shape/count identical between
Torch/NCCL and threshold32m. Backend selection is not currently recorded in
production stats because no production route hook was implemented.

## D2D Copy Accounting

The only D2D copies come from the PyNCCL symmetric all-reduce path:

- `[2496,4096]` BF16 input is about 19.5 MiB; copy-in plus copy-out is 39.0 MiB
  per all-reduce.
- `[1024,4096]` BF16 input is 8.0 MiB; copy-in plus copy-out is 16.0 MiB per
  all-reduce.
- `[16384,4096]` and `[9216,4096]` exceed 32M, so threshold32m routes them to
  direct PyNCCL all-reduce with no internal D2D buffer copies.
- PyNCCL all-gather writes directly to the output tensor, so the symmetric
  threshold does not add D2D copies for `lm_head_all_gather`.

## Graph Capture And Correctness

- Route microbench BF16/fp32 correctness: pass for all rows.
- Route microbench graph capture: pass for all rows.
- Partial `DistributedCommunicator` probe: pass for Torch/NCCL, PyNCCL direct,
  PyNCCL symmetric, and PyNCCL threshold32m.
- Full model threshold32m graph replay stayed zero-eager in all gate scenarios.
- Text smoke with threshold32m passed with graph replay/eager `9 / 0`.

Text smoke outputs:

- `2 + 2 等于 4。`
- `The sky is blue on a clear day.`
- `杭州：人间天堂，西湖美景。`

## Implementation Summary

No production runtime route hook was implemented.

Reason: the prompt's cheap-gate rule says not to enter full model or production
implementation when no-weight replay does not beat threshold32m. That is the
case here.

Experimental implementation only:

- extended `tp8_comm_backend_probe.py` with `pynccl_threshold32m`;
- added `torch_all`, `pynccl_threshold32m`,
  `route_small_hidden_to_pynccl`, `route_hidden_to_pynccl`, and
  `route_gather_to_pynccl` synthetic route policies;
- added prefix-like trace replay and route-selected backend/D2D-copy reporting.

Existing opt-in to use:

```bash
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 \
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M \
... --use-pynccl
```

No `MINISGL_COMM_ROUTE_POLICY` or route stats env flag was added.

## Promotion Decision

Decision: keep opt-in, do not promote, reject new owner/size routing for this
target.

Why keep threshold32m opt-in:

- Repeat gate was positive for the required repeated macro scenarios.
- Prefix health and text smoke passed.
- Graph replay stayed zero-eager.
- It reuses existing PyNCCL threshold logic with a simple rollback path.

Why not promote:

- No-weight replay still shows historical traces prefer Torch-like routing.
- Explicit owner/size policies did not improve over the global threshold on the
  serving/prefix replay where threshold32m is strongest.
- Owner timing/profile was intentionally not run after the route cheap gate
  rejected production routing.
- A full promotion pass would still need repeat-stable coverage including
  `historical_4096_128_bs4` in the same target.

## Next Steps

1. Keep the current opt-in documented for DSV4 SM80 TP8 experiments:
   `MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M --use-pynccl`.
2. If promotion is desired, run a dedicated repeat-stable global-threshold
   promotion gate across all four macro scenarios, including
   `historical_4096_128_bs4`, then run owner timing/profile.
3. Do not add owner routing until a cheaper replay shows a route beating
   `pynccl_threshold32m` on serving/prefix without giving back the historical
   stability margin.
4. If communication remains the bottleneck after the global threshold decision,
   make TARGET 10.3 about overlap/NCCL grouping/stream scheduling, or split a
   separate vLLM custom all-reduce port target. Do not mix that with this
   threshold routing target.

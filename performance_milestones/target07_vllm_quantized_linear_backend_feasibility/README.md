# TARGET 07.73 vLLM Quantized-Linear Backend Feasibility

Date: 2026-07-02

## Scope

This target tested standalone vLLM-aligned quantized-linear backends for the
DeepSeek V4 A100/sm80 decode-small projection shapes.

Baseline interpretation:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- promoted dense projection contract: mini activation FP8 rounding plus cached
  BF16 dequantized weights
- standalone runner: `/workspace/venvs/vllm-dsv4/bin/python`
- model: `/models/DeepSeek-V4-Flash`
- layer: `9`
- simulated TP rank: `0 / 8`

Inactive opt-ins intentionally not used as baseline:

```text
MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1
MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1
MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1
```

No mini runtime source files were changed.  This target stops at standalone
backend feasibility and a runtime integration plan.

## Artifacts

- `scripts/check_quantized_linear_backend_availability.py`
- `scripts/focused_quantized_linear_backend_microbench.py`
- `raw/backend_availability.json`
- `raw/focused_quantized_linear_backend_microbench.json`
- `raw/focused_quantized_linear_backend_microbench.md`
- `summaries/vllm_quantized_linear_backend_parity.md`
- `summaries/quantized_linear_roofline.md`

Commands:

```bash
/workspace/venvs/vllm-dsv4/bin/python -m py_compile \
  performance_milestones/target07_vllm_quantized_linear_backend_feasibility/scripts/check_quantized_linear_backend_availability.py \
  performance_milestones/target07_vllm_quantized_linear_backend_feasibility/scripts/focused_quantized_linear_backend_microbench.py

/workspace/venvs/vllm-dsv4/bin/python \
  performance_milestones/target07_vllm_quantized_linear_backend_feasibility/scripts/check_quantized_linear_backend_availability.py \
  --output performance_milestones/target07_vllm_quantized_linear_backend_feasibility/raw/backend_availability.json

/workspace/venvs/vllm-dsv4/bin/python \
  performance_milestones/target07_vllm_quantized_linear_backend_feasibility/scripts/focused_quantized_linear_backend_microbench.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 9 \
  --tokens 1 4 8 16 \
  --warmup 20 \
  --iters 80 \
  --output performance_milestones/target07_vllm_quantized_linear_backend_feasibility/raw/focused_quantized_linear_backend_microbench.json
```

## Availability

Full table: `summaries/vllm_quantized_linear_backend_parity.md`.

| Probe | Result |
| --- | --- |
| A100 device | `NVIDIA A100-SXM4-80GB`, capability `[8, 0]` |
| vLLM env | available from `/workspace/vllm-dsv4-docker` |
| mini env vLLM import | unavailable; use vLLM venv for standalone backend tests |
| DeepSeek V4 block FP8 | selects `MarlinFP8ScaledMMLinearKernel` |
| FBGEMM FP8 on A100 | routes to Marlin; activations are BF16 on Marlin path |
| Marlin W8A8 | rejected: `Marlin W8A8 is not supported.` |
| `torch._scaled_mm` FP8 | unsupported on A100 in this environment |
| INT8 W8A8 | selects `CutlassInt8ScaledMMLinearKernel` |

Important alignment decision: Lane A uses vLLM's weight-only Marlin contract.
It does not add activation FP8 quantization.  The promoted BF16 baseline still
includes mini's replay-time activation FP8 rounding, measured through the mini
Triton helper imported directly from `minisgl.kernel.triton.deepseek_v4`.

## Tested Owners

| Owner | Local shape | Notes |
| --- | --- | --- |
| attention WQA/WKV/compress | `N=1536, K=4096` | replicated fused WQA/WKV |
| attention `q_wqb` | `N=4096, K=1024` | TP8 column shard |
| attention `wo_b` local | `N=4096, K=1024` | TP8 row shard, all-reduce excluded |
| shared experts gate/up | `N=512, K=4096` | TP8 column shard of `w1/w3` concat |
| shared experts down | `N=4096, K=256` | TP8 row shard, all-reduce excluded |
| attention `wo_a` grouped | `N=1024, K=4096` | diagnostic as two Marlin/INT8 launches |

## Latency

Representative M=`4` results:

| Owner | Baseline ms | vLLM block Marlin ms | Speedup | FBGEMM-derived Marlin ms | INT8 W8A8 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| WQA/WKV/compress | `0.100167` | `0.077816` | `22.31%` | `0.077356` | `0.084380` |
| `q_wqb` | `0.092440` | `0.066223` | `28.36%` | `0.063482` | `0.083800` |
| `wo_b` local | `0.091272` | `0.064005` | `29.87%` | `0.063063` | `0.083567` |
| shared gate/up | `0.098032` | `0.079082` | `19.33%` | `0.079748` | `0.085368` |
| shared down | `0.092495` | `0.064094` | `30.70%` | `0.062869` | `0.083520` |
| `wo_a` grouped | `0.057180` | `0.189748` | `-231.84%` | `0.186664` | `0.230878` |

All-M gate read:

| Backend | Owners passing `>=15%` for M=`1,4,8,16` | Regressions `>10%` | Gate |
| --- | --- | --- | --- |
| `vllm_fp8_marlin_w8a16_block` | `q_wqb`, `wo_b local`, shared experts down | none for dense backend | pass |
| `vllm_fbgemm_fp8_marlin_derived_channel` | `q_wqb`, `wo_b local`, shared experts down | none for dense backend | pass, but diagnostic |
| `vllm_int8_w8a8_cutlass_dynamic` | none | none for dense backend | fail |
| grouped `wo_a` two-launch variants | none | `wo_a` grouped | fail |

Lane A is the preferred passing lane because it preserves the native DeepSeek
V4 block FP8 `weight_scale_inv` contract.  Lane B passes timing but requires a
load-time block-FP8 to BF16 to per-channel-FP8 conversion and should not be the
first integration target.

## Quality

M=`4` error versus promoted cached BF16 outputs:

| Owner | Block Marlin max abs | Block Marlin mean abs | Block Marlin cosine | FBGEMM-derived mean abs | INT8 mean abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| WQA/WKV/compress | `0.179688` | `0.033776` | `0.99964100` | `0.046860` | `0.036730` |
| `q_wqb` | `0.210938` | `0.016412` | `0.99964738` | `0.023972` | `0.017910` |
| `wo_b` local | `0.125000` | `0.015123` | `0.99963570` | `0.022179` | `0.016207` |
| shared gate/up | `0.109375` | `0.023698` | `0.99962711` | `0.033299` | `0.026601` |
| shared down | `0.062500` | `0.005558` | `0.99966508` | `0.007541` | `0.005920` |
| `wo_a` grouped | `0.007812` | `0.000002` | `0.99999994` | `0.025911` | `0.017143` |

The dense Marlin errors are primarily from changing the activation policy from
mini's promoted activation FP8 rounding to vLLM's BF16-input weight-only
Marlin path.  This is acceptable for standalone feasibility, but a runtime
opt-in must run TP8 text smoke and hidden/logit checks before any macro claim.

## Preparation And Memory

Representative one-layer prep and persistent bytes:

| Owner | Backend | Prep ms | Conversion ms | Persistent bytes | Workspace bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| WQA/WKV/compress | promoted cached BF16 | `31.267` | `31.267` | `12,582,912` | `0` |
| WQA/WKV/compress | block Marlin | `52.955` | `0.000` | `6,390,192` | `432` |
| `q_wqb` | promoted cached BF16 | `0.397` | `0.397` | `8,388,608` | `0` |
| `q_wqb` | block Marlin | `0.720` | `0.000` | `4,260,272` | `432` |
| `wo_b` local | promoted cached BF16 | `1.716` | `1.716` | `8,388,608` | `0` |
| `wo_b` local | block Marlin | `0.693` | `0.000` | `4,260,272` | `432` |
| shared gate/up | promoted cached BF16 | `0.175` | `0.175` | `4,194,304` | `0` |
| shared gate/up | block Marlin | `0.662` | `0.000` | `2,130,352` | `432` |
| shared down | promoted cached BF16 | `0.150` | `0.150` | `2,097,152` | `0` |
| shared down | block Marlin | `0.634` | `0.000` | `1,065,392` | `432` |
| `wo_a` grouped | promoted cached BF16 | `0.910` | `0.910` | `8,388,608` | `0` |
| `wo_a` grouped | block Marlin two-launch | `1.157` | `0.000` | `4,260,704` | `864` |

Prep includes one-time vLLM block processing, Marlin repack, scale expansion,
scale permutation, and exponent-bias fusion where applicable.  No candidate
timing includes repeated repacking.

## Roofline

Full table: `summaries/quantized_linear_roofline.md`.

Readout:

- These M=`1..16` projections are launch/backend-bound, not close to raw
  compute or HBM lower bounds.
- Marlin still gives a useful dense-owner standalone signal by reducing replay
  weight/scale traffic and matching vLLM's no-activation-quant SM80 policy.
- The grouped `wo_a` two-launch diagnostic proves that lower weight bytes alone
  are not enough.
- Since Lane A already passes standalone gates, open a runtime opt-in target
  before custom-kernel R&D.

## Decision

Decision: standalone backend gate passes for a bounded dense-owner subset.

Selected backend for the next target:

```text
vLLM FP8 Marlin W8A16 block linear
```

First integration subset:

- `attn.q_wqb`
- `attn.wo_b` local projection
- shared experts down

Optional expansion after first profile:

- WQA/WKV/compress
- shared experts gate/up

Rejected in this target:

- `wo_a` grouped two-launch Marlin/FBGEMM/INT8 variants
- `torch._scaled_mm` FP8 on A100
- INT8 W8A8 projection as a silent runtime change
- FBGEMM-derived conversion as the first integration route

No runtime implementation was added in 07.73.

## Next Target Recommendation

Open a bounded runtime opt-in target, for example:

```text
TARGET 07.74: DSV4 SM80 vLLM FP8 Marlin Dense Projection Runtime Opt-In
```

Suggested toggle:

```text
MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1
```

Required gates for that target:

- build Marlin-repacked weights/scales before CUDA graph capture;
- no per-decode repack, scale conversion, or workspace allocation;
- first wire only `q_wqb`, `wo_b local`, and shared experts down;
- verify hidden/logit quality and TP8 text smoke before macro;
- preserve graph replay and eager decode `0`;
- fresh 4096/128 Nsight profile must show a projection/GEMM reduction large
  enough to plausibly move macro, with a first-subset floor around `0.04s` and
  an expanded-subset floor around `0.07s`;
- 4096/1024 macro promotion still requires a same-run output tok/s gain and no
  quality regression.

## Do Not Continue

Do not continue the `wo_a` two-launch Marlin route.  Do not treat FBGEMM
per-channel conversion as the native DeepSeek V4 path.  Do not pursue
`torch._scaled_mm` FP8 on A100 unless the environment/backend support changes.
Do not promote INT8 projection without a separate precision and quality target.

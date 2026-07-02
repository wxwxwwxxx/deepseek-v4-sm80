# HC/Router Precision Probe

## Scope

This probe is measurement-only.  It did not change runtime code and did not
promote any opt-in.  The current baseline remains:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- page size: `256`
- TP8, 8x A100 for macro/profile interpretation

Probe command:

```bash
CUDA_VISIBLE_DEVICES=0 MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
python performance_milestones/target07_precision_boundary_pivot/scripts/hc_router_precision_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 9 \
  --tokens 1 4 8 16 \
  --warmup 50 \
  --iters 200 \
  --output performance_milestones/target07_precision_boundary_pivot/raw/hc_router_precision_probe.json
```

Quality supplement:

```bash
CUDA_VISIBLE_DEVICES=0 MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
python performance_milestones/target07_precision_boundary_pivot/scripts/hc_router_precision_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --layer 9 \
  --tokens 1024 \
  --warmup 10 \
  --iters 30 \
  --output performance_milestones/target07_precision_boundary_pivot/raw/hc_router_precision_probe_rows1024.json
```

Raw artifacts:

- `raw/hc_router_precision_probe.json`
- `raw/hc_router_precision_probe.md`
- `raw/hc_router_precision_probe_rows1024.json`
- `raw/hc_router_precision_probe_rows1024.md`

## Decode-Small Probe

`M=4` is used as the decode-batch proxy for the profile-gain estimate.

| Owner/case | FP32 mean ms | TF32 mean ms | TF32 delta | BF16-like mean ms | BF16-like delta | Output/top-k risk |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HC attn pre linear | `0.059001` | `0.061749` | `+4.66%` | `0.041540` | `-29.59%` | BF16-like max abs err `0.039168`; TF32 max abs err `0.005124` |
| HC ffn pre linear | `0.058856` | `0.060716` | `+3.16%` | `0.041794` | `-28.99%` | BF16-like max abs err `0.044204`; TF32 max abs err `0.003016` |
| MoE router gate linear | `0.070008` | `0.071657` | `+2.36%` | `0.043721` | `-37.55%` | M=4 top-k overlap `1.000000` for both TF32 and BF16-like |

Profile-gain estimate against TARGET 07.69 owners:

| Variant | HC pre estimate | Router estimate | Combined estimate | Gate read |
| --- | ---: | ---: | ---: | --- |
| TF32-enabled | `0.000000s` | `0.000000s` | `0.000000s` | Fails `>=0.05s`; decode-small TF32 is slower/noisy negative. |
| BF16-like | `0.052250s` | `0.036464s` | `0.088714s` | Has theoretical speed, but changes precision and router decisions. |

## Router Quality Supplement

The `M=1024` run is not used as a decode-latency proxy.  It is a larger-row
router stability probe.

| Variant | Router mean ms | Max abs err | Mean abs err | Top-k set overlap | Exact order match | Changed rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF32-enabled | `0.104015` | `0.0000439` | `0.00000480` | `1.000000` | `1.000000` | `0 / 1024` |
| BF16-like | `0.048832` | `0.0596504` | `0.0052975` | `0.992350` | `0.854492` | `149 / 1024` |

## Readout

- TF32 is the only low-risk local precision change tested here, but it does
  not reduce the decode-small HC/router surface.  It fails the `0.05s`
  credible profile-gain stop rule.
- BF16-like HC/router has a theoretical microbench speedup, but the router
  precision probe changes top-k routing on the larger-row sample.  There is no
  clear quality path for accepting that inside this short target.
- HC-only BF16-like would be roughly `0.052s` theoretical on the 07.69 HC
  owner, but this is barely above the stop threshold, changes HC outputs, and
  is not vLLM's SM80 HC contract.  vLLM `mhc_pre` keeps the HC `fn` and
  post/comb contract in FP32-like precision.

Decision: do not choose exact-ish HC/router as the next implementation lane.

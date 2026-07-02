# TARGET 07.71 Precision / Boundary Pivot

Date: 2026-07-02

## Scope

This target is a short measurement and decision pivot.  It did not implement
full FP8 projection/cache, full `fp8_ds_mla`, full-model compile/runtime
rewrite, or any opt-in promotion.

Baseline for interpretation:

- variant: `dsv4_sm80_a100_victory`
- env: `MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1`
- page size: `256`
- TP8, 8x A100

Inactive opt-ins that remain outside the baseline:

- `MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`
- `MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1`
- `MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1`

Repo status at report time:

```text
## dsv4-sglang-based...origin/main [ahead 44]
?? performance_milestones/target07_precision_boundary_pivot/
```

## Artifacts

- `scripts/hc_router_precision_probe.py`
- `raw/hc_router_precision_probe.json`
- `raw/hc_router_precision_probe.md`
- `raw/hc_router_precision_probe_rows1024.json`
- `raw/hc_router_precision_probe_rows1024.md`
- `summaries/hc_router_precision_probe.md`
- `summaries/pivot_candidate_table.md`
- `summaries/vllm_precision_boundary_parity.md`

## TARGET 07.70 Negative Result Summary

TARGET 07.70 tested the exact-route BF16 small-GEMM pretranspose path:

- variant: `dsv4_sm80_a100_victory_bf16smallgemm`
- toggle: `MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1`
- behavior: build pretransposed BF16 cached weights before CUDA graph capture
  and route decode-small `rows <= 16` GEMMs through `torch.mm(x_2d, weight_t)`
- covered owners: WQA/WKV/compress, `attn.q_wqb`, `attn.wo_b`,
  `indexer.wq_b`, shared expert gate/up, shared expert down
- not changed: HC pre linear, MoE router FP32/SGEMM, `attn.wo_a` grouped BMM

The focused microbench gate passed, with representative M=4 speedups around
`15%-18%`, and TP8 text smoke passed.  But the profile and macro gates failed:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Projection/GEMM bucket | `0.778887s` | `0.778170s` | `-0.000717s` |
| BF16 cluster | `0.521619s` | `0.521012s` | `-0.000607s` |
| cuBLASLt BF16 GEMM | `0.219912s` | `0.313612s` | `+42.61%` |
| CUTLASS BF16 GEMM | `0.194319s` | `0.095964s` | `-50.62%` |
| cuBLASLt splitK/reduce | `0.107388s` | `0.111436s` | `+3.77%` |
| 4096/128 output tok/s | `62.3274` | `62.3750` | `+0.08%` |
| 4096/1024 output tok/s | `131.7927` | `131.9084` | `+0.09%` |

Interpretation: BF16 pretranspose changed the backend-family mix but did not
reduce aggregate projection/GEMM or macro time.  Do not continue narrow BF16
layout polishing without new evidence.

## HC/Router Precision Probe

`M=4` decode-small probe:

| Owner/case | FP32 mean ms | TF32 mean ms | TF32 delta | BF16-like mean ms | BF16-like delta | Quality note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HC attn pre linear | `0.059001` | `0.061749` | `+4.66%` | `0.041540` | `-29.59%` | BF16-like max abs err `0.039168` |
| HC ffn pre linear | `0.058856` | `0.060716` | `+3.16%` | `0.041794` | `-28.99%` | BF16-like max abs err `0.044204` |
| MoE router gate linear | `0.070008` | `0.071657` | `+2.36%` | `0.043721` | `-37.55%` | M=4 top-k overlap `1.000000` |

Profile-gain estimate from TARGET 07.69 owners:

| Variant | HC pre estimate | Router estimate | Combined estimate | Decision |
| --- | ---: | ---: | ---: | --- |
| TF32-enabled | `0.000000s` | `0.000000s` | `0.000000s` | Stop: no credible `>=0.05s` gain. |
| BF16-like | `0.052250s` | `0.036464s` | `0.088714s` | Stop: precision risk and router movement. |

Router quality supplement at `M=1024`:

| Variant | Max abs err | Mean abs err | Top-k set overlap | Exact order match | Changed rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF32-enabled | `0.0000439` | `0.00000480` | `1.000000` | `1.000000` | `0 / 1024` |
| BF16-like | `0.0596504` | `0.0052975` | `0.992350` | `0.854492` | `149 / 1024` |

Readout:

- TF32 is quality-stable but gives no decode-small speedup.
- BF16-like has speed in microbench but visibly changes router top-k behavior
  on the larger-row probe.
- HC-only BF16-like is only barely above the `0.05s` theoretical threshold and
  changes HC outputs; it is not vLLM's SM80 HC contract.

Do not select exact-ish HC/router as the next implementation target.

## vLLM FP8 / Custom-Boundary Ranking

| Rank | Candidate mechanism | Mini owner/cluster | Expected gain | Quality risk | Engineering size | Decision |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | vLLM-aligned FP8/custom projection-cache cluster | BF16 small-GEMM cluster `0.521012s` (`0.521619s` baseline) | `0.12s-0.20s` target; require at least `0.10s` profile reduction | Medium, measurable by per-owner error and TP8 text smoke | Medium-large | Select as next target. |
| 2 | WQA/WKV-only FP8 fused projection | WQA/WKV/compress `0.123405s` | Below `0.20s` standalone gate | Medium | Medium | Use as a representative inside rank 1, not standalone. |
| 3 | `fused_inv_rope_fp8_quant` + `deepseek_v4_fp8_einsum` | `wo_a` `0.064244s` + `wo_b` local `0.051837s` | Below current gate | Medium-high | Medium | Hold; 07.62 already collapsed `wo_a`. |
| 4 | Fused indexer q/rope/quant and packed FP8 indexer/cache continuation | indexer projection pieces about `0.084222s`; prior FP8 indexer pieces already promoted | Below current gate | Medium | Medium | Hold unless a fresh profile reselects indexer/cache. |
| 5 | Full `fp8_ds_mla` KV-cache E2E | Attention/cache architecture, not isolated by this pivot | Potentially large but not isolatable | High | Large | Do not select as next small target. |
| 6 | Whole-model compile/custom-op boundary | No concrete owner/cluster mapping from this target | Unknown | Medium | Large | Do not select; broad compile is out of scope. |

Source parity details are in `summaries/vllm_precision_boundary_parity.md`.

## Quality-Risk Notes

- Router precision is high-risk.  vLLM requests FP32 router logits, and the
  BF16-like mini probe changed top-k routing.  Do not pursue BF16 router
  without a separate quality-policy target.
- HC BF16-like is also a precision-policy change.  The theoretical HC-only
  gain is too close to the threshold to justify quality risk here.
- FP8/custom projection-cache work is allowed only as an opt-in implementation
  target with explicit gates.  It must compare against the promoted BF16 path,
  preserve graph replay, keep eager decode at `0`, and run TP8 text smoke
  before any macro claim.
- Full FP8 KV-cache E2E and whole-model compile remain out of scope for the
  next small target.

## Do Not Continue

- Do not continue narrow BF16 layout/pretranspose polishing from TARGET 07.70.
- Do not continue HC/router TF32: no credible decode-small profile gain.
- Do not continue BF16-like router precision: larger-row top-k routing changed
  and no quality path is defined.
- Do not choose standalone `wo_a` FP8-einsum: current promoted `wo_a` surface
  is too small after 07.62.
- Do not choose standalone indexer/cache continuation unless a fresh profile
  reselects it.
- Do not choose full `fp8_ds_mla` or whole-model compile as the next small
  target, because isolated value cannot be verified without a large E2E route.

## Decision

Decision:

- Next target: `TARGET 07.72 DSV4 SM80 vLLM-Aligned FP8/Custom Projection-Cache Boundary`
- Selected lane: vLLM-aligned FP8/custom boundary
- Target owner/cluster: current projection-cache BF16 small-GEMM cluster,
  `0.521012s-0.521619s`, including WQA/WKV/compress, shared experts,
  `wo_a`, `q_wqb`, `wo_b`, and indexer projection pieces
- Expected profile gain: target `0.12s-0.20s` rank0 decode-envelope reduction;
  require at least `0.10s` before any promotion discussion
- Expected macro gain: `+3%-6%` 4096/1024 output tok/s if profile reduction is
  at least `0.12s` and graph replay remains active
- Quality gate: per-owner output error vs promoted BF16 path
  (`max/mean/p99 abs` and relative), attention/logit smoke where applicable,
  TP8 text smoke pass, graph replay nonzero, eager decode `0`, no opt-in
  promotion in the implementation target
- Why not the other lanes: TF32 HC/router has no decode-small speedup; BF16
  router changes top-k routing; standalone `wo_a` and indexer candidates are
  below current owner/cluster gates; full `fp8_ds_mla` and whole-model compile
  require too much E2E machinery before isolated value can be validated
- Stop condition for the next target: stop without runtime implementation if
  source parity cannot map the custom FP8 projection boundary to at least
  `0.35s` of coherent mini surface; stop without macro if real-shape focused
  microbench does not show at least `15%` latency reduction on at least two
  representative owners; stop without promotion if fresh 4096/128 profile
  reduction is below `0.10s` or same-run 4096/1024 macro gain is below `3%`

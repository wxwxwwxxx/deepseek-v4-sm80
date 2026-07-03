# TARGET 07.77: Dense FP8 Marlin Runtime Regression Attribution

Date: 2026-07-03

Status: diagnostic complete.  Do not promote dense FP8 Marlin projection, do
not expand Phase B owners, and do not start a new optimization lane from the
07.76 single-run regression.

## Measurement Method

The current perf-matrix harness still constructs one `LLM`/Engine per torchrun.
Because per-variant env lifecycle is the known same-run blocker, this target
kept separate baseline/candidate invocations and used repeated generations to
test stability.

Artifacts:

- repeated clean macro: `raw/repeat2_4096x1024_*_np128/`;
- owner timing profile: `raw/timing_4096x128_*_np128/` and
  `raw/timing_4096x1024_*_np128/`;
- summary: `summaries/target0777_summary.json`;
- commands: `scripts/run_commands.md`.

The owner timing instrumentation is gated by `MINISGL_DSV4_OWNER_TIMING=1`.
Default-disabled paths have fast paths to avoid affecting normal macro runs.
Timing runs are diagnostic only: CUDA events are captured into the decode graph
and host prepare substeps are timed, so their throughput is not a promotion
gate.

## Fairness / Repeatability

4096/1024, batch 4, TP8, page size 256, `np128`, two repeats per fresh Engine:

| Variant | Output tok/s | Decode tok/s | TTFT s | Elapsed s | Decode forward s | Prepare s | Replay | Eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `126.1360` | `168.6624` | `6.2615` | `64.9458` | `48.5230` | `6.8859` | `2046` | `0` |
| candidate | `123.7841` | `165.9067` | `6.4772` | `66.1797` | `49.3289` | `5.2226` | `2046` | `0` |
| delta | `-1.86%` | `-1.63%` | `+0.2156` | `+1.2340` | `+0.8060` | `-1.6633` | same | same |

Per-repeat evidence shows the old 07.76 `prepare_s +1.7s` pattern is not
stable:

| Repeat | Baseline output tok/s | Candidate output tok/s | Baseline TTFT | Candidate TTFT | Main difference |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | `121.2268` | `116.9170` | `7.5688` | `7.9849` | candidate prefill forward was `+2.10s`; baseline prepare was `+1.72s` higher |
| 1 | `131.4680` | `131.5082` | `4.9542` | `4.9694` | effectively neutral |

So the 07.76 single-run `-6.70%` long macro regression reproduced only as a
smaller, first-repeat-sensitive `-1.86%` aggregate here.  The stable second
repeat is neutral.

## Prepare / TTFT Breakdown

The 07.76 regression reported `prepare_s: 2.6811 -> 4.4067` and
`TTFT: 5.8932 -> 7.6155`.  That specific prepare/TTFT increase did not
reproduce.

In the 4096/1024 owner-timing runs, host prepare substeps were nearly identical:

| Substep, rank-max host ms | Baseline | Candidate |
| --- | ---: | ---: |
| prefill attention metadata | `777.63` | `760.86` |
| prefill allocate pages | `32.79` | `21.39` |
| decode attention metadata total | `1605.29` | `1603.15` |
| decode positions total | `62.45` | `62.39` |
| decode write tuple total | `50.58` | `50.07` |

Dense FP8 Marlin model prepare is a cold-init cost, not part of measured
generation elapsed.  Candidate rank-max host time for all dense Marlin prepare
calls was `914.03 ms`; CUDA-op rank-max subtotal was about `139.14 ms`, led by:

| Dense Marlin prepare CUDA op | Rank-max ms |
| --- | ---: |
| scale exponent-bias fusion | `62.51` |
| scale permute | `33.76` |
| scale cast | `12.21` |
| pack FP8 to int32 | `7.23` |
| GPTQ Marlin repack | `6.30` |

This can matter for cold-start/load-init accounting, but it does not explain a
4096/1024 output-throughput regression inside `generate()`.

## Owner-Level Runtime Timing

4096/1024 timing below is rank-max, one captured bs4 decode replay, summed over
43 layers, then estimated over `1023` replays.

| Owner | Baseline local ms/replay | Candidate local ms/replay | Est. delta s / 1023 | Pure GEMM/custom-op delta s / 1023 |
| --- | ---: | ---: | ---: | ---: |
| `attn.q_wqb` | `1.8045` | `1.8140` | `+0.0098` | `-0.0920` |
| `attn.wo_b` local | `1.7297` | `1.7717` | `+0.0429` | `-0.0531` |
| `shared_experts.down_proj` local | `1.5373` | `1.5784` | `+0.0420` | `-0.0589` |

Dense Marlin pure GEMM/custom-op was faster for all three owners.  The local
total is only slightly worse because the diagnostic graph contains event ranges
around view boundaries; layout counters below show those boundaries are not real
copies.

## Layout / Copy Attribution

4096/1024 candidate layout counters across ranks:

| Counter | Count |
| --- | ---: |
| `q_wqb` input reshape view | `2408` |
| `q_wqb` output reshape view | `2408` |
| `q_wqb` contiguous skipped | `2408` |
| `wo_b` input reshape view | `2408` |
| `wo_b` output reshape view | `2408` |
| `wo_b` contiguous skipped | `2408` |
| `shared_down` input reshape view | `2408` |
| `shared_down` output reshape view | `2408` |
| `shared_down` contiguous skipped | `2408` |

No dense Marlin owner took the `.contiguous()` branch in the captured decode
shape, and no input/output reshape copy was observed.  Layout/copy is therefore
not the regression bucket.

## All-Reduce Timing / Ordering

Communication counters stayed identical in the 4096/1024 timing runs:

| Label | Count | Bytes |
| --- | ---: | ---: |
| `dsv4.attn.wo_b.row_parallel_projection_all_reduce` | `344` | `46,170,898,432` |
| `dsv4.v1_moe_reduce_once_all_reduce` | `344` | `92,341,796,864` |
| total communication | `704` | `139,602,984,960` |

Rank-max all-reduce timing:

| All-reduce | Baseline ms/replay | Candidate ms/replay | Est. delta s / 1023 |
| --- | ---: | ---: | ---: |
| `wo_b` row-parallel | `2.5067` | `2.6003` | `+0.0958` |
| MoE reduce-once | `2.2080` | `2.2061` | `-0.0019` |

Ordering stayed equivalent at the owner level:

```text
baseline:  q_wqb -> wo_b local -> wo_b all_reduce -> shared_down -> moe reduce_once
candidate: q_wqb Marlin -> wo_b Marlin -> wo_b all_reduce -> shared_down Marlin -> moe reduce_once
```

There is a small `wo_b` all-reduce timing movement, but no byte/count growth and
no ordering inversion large enough to explain 07.76's `+2.31s` elapsed loss.

## 4096/128 Profile

The short timing profile also kept graph replay active (`127`) and eager decode
at `0`.  Estimated owner deltas over 127 replays were small:

| Owner | Est. local delta s / 127 | Pure GEMM/custom-op delta s / 127 |
| --- | ---: | ---: |
| `attn.q_wqb` | `-0.0068` | `-0.0158` |
| `attn.wo_b` local | `+0.0247` | `-0.0091` |
| `shared_down` local | `+0.0022` | `-0.0062` |
| `wo_b` all-reduce | `+0.0341` | n/a |

This matches the 07.76 short macro behavior: no meaningful regression at the
short shape, and no layout-copy issue.

## Regression Classification

Primary bucket: `measurement fairness/noise`.

Evidence:

- the 07.76 long regression pattern (`prepare_s +1.7s`, `TTFT +1.7s`) did not
  reproduce under repeated clean runs;
- the second repeat was neutral: `131.4680 -> 131.5082 output tok/s`;
- pure dense Marlin GEMM/custom-op timing improved for `q_wqb`, `wo_b`, and
  `shared_down`;
- all dense Marlin layout boundaries were views, and `.contiguous()` was always
  skipped for the captured decode shape;
- communication bytes/counts were unchanged, with only a small `wo_b`
  all-reduce timing shift of about `0.096s` over the long decode.

Secondary bucket: small mixed steady-state/ordering noise.  The measurable
owner-level candidate losses are about `0.19s` over 1023 replays
(`q_wqb`/`wo_b`/`shared_down` local totals plus `wo_b` all-reduce), far smaller than
the original 07.76 `+2.31s` elapsed loss and not stable across repeats.

## Decision

Keep dense FP8 Marlin projection behind
`MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1`.

Do not promote into `dsv4_sm80_a100_victory` from this evidence, but also do
not treat the 07.76 single-run long macro as a kernel/backend regression.  The
macro gate needs a fair lifecycle/repeat policy before this opt-in can be judged
again.

## Next Target

Recommended next target: benchmark lifecycle and first-repeat stabilization for
`dsv4_sm80_a100_victory` vs `dsv4_sm80_a100_victory_densefp8marlinproj`.

The largest seconds-scale loss is not an owner kernel; it is first-repeat
prefill/TTFT variance.  The next target should construct a fresh Engine per
variant in one controlled harness or make `warmup_repeats >= 1` plus repeated
macro summaries the promotion gate.  Only after that should any owner-level
optimization target be selected.

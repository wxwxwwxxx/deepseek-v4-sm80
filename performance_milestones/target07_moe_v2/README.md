# TARGET 07.3 MoE V2 Exact

Status: first implementation cut in progress.

## Scope

TARGET 07.3 focuses only on exact DeepSeek V4 MoE execution on sm80. The path
stays bf16-direct for activations, keeps FP4 expert weights as today, and does
not add INT8, activation quantization, MXFP4/FP8 semantic changes, or a vLLM
runtime dependency.

The first cut mirrors the useful vLLM FusedMoE boundaries without porting
`DeepseekV4MegaMoEExperts`:

- route metadata is owned by a `DSV4MoEExecutionPlan`
- grouped MoE temporary tensors are owned by a per-layer `DSV4MoEWorkspace`
- routed and shared experts keep the V1 reduce-once boundary
- dispatch/finalize remain inside mini Triton wrappers

## First Cut

Toggle: `MINISGL_DSV4_SM80_MOE_V2=1`

Selectable variants:

- `v1_moe_v2`
- `v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache`

Expected behavior:

- V2 implies the V1 exact MoE route whitelist.
- The gate still produces fp32 route weights and integer expert ids.
- `DSV4MoE.forward` builds one execution plan after gating.
- Routed expert dispatch consumes the plan and uses a reusable workspace for
  decode-sized route counts only. Large prefill keeps ephemeral temporaries so a
  per-layer workspace does not retain prefill-sized W2 routed buffers.
- Routed and shared expert outputs are summed before the tensor-parallel reduce.

## Correctness

Run targeted unit coverage:

```bash
pytest -q -o addopts='' \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

Run a model text smoke for the new exact V2 macro variant:

```bash
python benchmark/offline/deepseek_v4_text_smoke.py \
  --variants v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --output performance_milestones/target07_moe_v2/text_smoke_v2.json
```

Correctness criteria:

- no text sanity failures
- no new fallback to non-exact activation quantization
- no `MINISGL_DSV4_SM80_MOE_INT8` or precision-lane toggles in raw env
- V2 grouped output matches the existing bf16 reference tolerance in microbench

## Microbench

Default synthetic cases:

```bash
python benchmark/offline/deepseek_v4_moe_route_microbench.py \
  --warmup 5 \
  --iters 20 \
  --output performance_milestones/target07_moe_v2/moe_v2_microbench.json
```

DSV4-like shape artifact:

```bash
python benchmark/offline/deepseek_v4_moe_route_microbench.py \
  --include-real-shapes \
  --warmup 5 \
  --iters 20 \
  --output performance_milestones/target07_moe_v2/moe_v2_microbench_real_shapes.json
```

Track:

- `route_metadata_ms`
- `v2_plan_ms`
- `v1_grouped_full_ms`
- `v2_grouped_full_ms`
- `v2_grouped_dispatch_ms`
- `v2_vs_v1_grouped_max_abs`

## Macro

Run the 4096/1024/batch4 closure scenario against the prior best and V2:

```bash
python benchmark/offline/deepseek_v4_perf_matrix.py \
  --variants \
    v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
    v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
  --scenarios mixed_prefill_decode_bs4 \
  --output-dir performance_milestones/target07_moe_v2/macro_4096_1024_bs4
```

Record output tok/s, TTFT, decode forward time, kernel counters, and fallback
wrapper totals in this directory.

## Profile

Capture an nsys artifact for the V2 macro variant:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=performance_milestones/target07_moe_v2/nsys_moe_v2_macro \
  python benchmark/offline/deepseek_v4_perf_matrix.py \
    --variants v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache \
    --scenarios mixed_prefill_decode_bs4 \
    --output-dir performance_milestones/target07_moe_v2/nsys_macro_report
```

Summarize W13/W2 grouped kernel time, route metadata time, route sum/finalize
time, shared expert overlap or serialization, and the final all-reduce boundary.

## Stop Rules

Stop 07.3 and move to `prompts/TARGET_07.35_dsv4_sm80_post_moe_reparity.md`
when any of these is true:

- one serious MoE cut improves 4096/1024 macro by at least 1.3x
- W13 plus W2 summed kernel time drops by at least 2x
- a fresh profile shows MoE is no longer a top-two bottleneck
- the next obvious step belongs to attention/cache, communication, precision, or
  ordinary graph cleanup

Stop this target and record the reason if two consecutive MoE cuts each produce
less than 5 percent macro gain and less than 10 percent routed-MoE subgraph gain.

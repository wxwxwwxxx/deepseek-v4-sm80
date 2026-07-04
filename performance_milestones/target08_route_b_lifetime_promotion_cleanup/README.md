# TARGET 08.29 Route B Lifetime Promotion Cleanup

## Exact Files Changed

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `tests/benchmark/test_deepseek_v4_perf_matrix.py`
- `tests/benchmark/test_deepseek_v4_text_smoke.py`
- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.30_dsv4_sm80_post_prefix_reprofile_next_bottleneck.md`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/README.md`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/scripts/run_promoted_routeb_lifetime_text_smoke.sh`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/raw/.gitkeep`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/raw/text_smoke_promoted_routeb_lifetime_verify.json`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/raw/text_smoke_promoted_routeb_lifetime_verify.dsv4_sm80_a100_victory_prefix_routeb_lifetime.json`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/raw/text_smoke_promoted_routeb_lifetime_verify.log`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/summaries/.gitkeep`

No runtime, scheduler, cache, attention, MoE, NCCL, low-precision, SWA
ownership, raw metadata graph-prep, or reference-assign mechanism was added.

## Promoted Variant

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

This is the promoted Route B lifetime prefix preset for TARGET 08.30.  It is
available in both:

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

## Env And CLI Composition

Variant env:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
```

Required invocation shape:

```bash
--variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
--page-size 256 \
--num-pages 128 \
--enable-dsv4-radix-prefix-cache \
--enable-dsv4-component-loc-ownership \
--allow-dsv4-cuda-graph \
--cuda-graph-bs 1 2 4 8 16
```

The variant keeps `allow_dsv4_cuda_graph=True` and
`cuda_graph_capture_greedy_sample=True`, matching the existing promoted A100
victory graph behavior.  The page-size, page-count, prefix-cache, component
ownership, and graph-bucket flags remain explicit CLI settings so existing
benchmark harness behavior is not changed globally.

## Verifier Preservation

Both perf matrix and text smoke preserve:

```text
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1
```

across variant env reset.  The reset still clears unrelated
`MINISGL_DSV4_SM80_*` variables, applies the selected variant env, and then
restores the verifier env.  The tiny TP8 smoke confirmed that the promoted
variant's raw env included the verifier after reset.

## Alias Kept

The old diagnostic variant remains available in both benchmark entry points:

```text
dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime
```

It is now a historical alias with the same env and graph flags as the promoted
variant.  Existing TARGET 08.27/08.28 artifacts and scripts can still be
reproduced with the old name, while new docs point to
`dsv4_sm80_a100_victory_prefix_routeb_lifetime`.

## Tests And Smoke Commands

Compile check:

```bash
python -m py_compile \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  python/minisgl/attention/deepseek_v4.py \
  python/minisgl/kernel/deepseek_v4.py
```

Result: pass.

Unit tests:

```bash
pytest -q \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/attention/test_deepseek_v4_backend_metadata.py
```

Result: `64 passed in 10.08s`.

Tiny TP8 text smoke with verifier:

```bash
performance_milestones/target08_route_b_lifetime_promotion_cleanup/scripts/run_promoted_routeb_lifetime_text_smoke.sh
```

Result: pass.

Smoke summary:

| field | value |
| --- | --- |
| status | pass |
| variant | `dsv4_sm80_a100_victory_prefix_routeb_lifetime` |
| outputs | `杭州西湖位于杭州市。`; `浙江省。`; `Blue.` |
| verifier preserved | `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1` |
| graph replay/eager | `5/0` |
| requested graph buckets | `[1, 2, 4, 8, 16]` |

Artifacts:

- `raw/text_smoke_promoted_routeb_lifetime_verify.json`
- `raw/text_smoke_promoted_routeb_lifetime_verify.dsv4_sm80_a100_victory_prefix_routeb_lifetime.json`
- `raw/text_smoke_promoted_routeb_lifetime_verify.log`

## Final Decision

`ready for TARGET 08.30`

TARGET 08.30 should use
`dsv4_sm80_a100_victory_prefix_routeb_lifetime` for the global post-prefix
reprofile.  The cleanup stays an opt-in preset and does not make prefix cache
the global default.

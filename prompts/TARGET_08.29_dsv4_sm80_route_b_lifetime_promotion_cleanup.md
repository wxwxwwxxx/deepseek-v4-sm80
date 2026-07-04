# TARGET 08.29: DSV4 Route B Lifetime Promotion Cleanup

## Status

Run this after TARGET 08.28 and before TARGET 08.30.

TARGET 08.28 promoted the TARGET 08.27 Route B component page-table lifetime
cache as the preferred Route B prefix-cache opt-in.  Before the global
post-prefix reprofile, clean up the public benchmark/text-smoke entry points so
future targets can use one clear promoted prefix configuration instead of
manually composing several environment variables.

This is intentionally a small cleanup target.  Do not add a new optimization
mechanism here.

## Goal

Make the promoted Route B prefix-cache path easy, explicit, and reproducible.

The promoted prefix path should include:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--page-size 256
--num-pages 128
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Keep it as an opt-in/preset, not as the global default for all DSV4 runs.

## Required Reading

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.28_dsv4_sm80_route_b_lifetime_cache_promotion_gate.md`
- `performance_milestones/target08_route_b_lifetime_cache_promotion_gate/README.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/README.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/DESIGN.md`

Core code references:

- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/server/args.py`
- `tests/benchmark/test_deepseek_v4_perf_matrix.py`
- `tests/benchmark/test_deepseek_v4_text_smoke.py`
- `tests/attention/test_deepseek_v4_backend_metadata.py`

## Scope

Allowed:

- add or rename benchmark/text-smoke variants for the promoted Route B lifetime
  path;
- preserve verifier env behavior for smoke and perf-matrix variants;
- update help strings, docs, and route files to make the promoted prefix path
  unambiguous;
- add focused unit tests for variant env composition and verifier preservation;
- add a short smoke script under `performance_milestones`;
- optionally run one short correctness/text smoke and one tiny graph replay
  sanity check.

Not allowed:

- changing model/math precision;
- changing SWA ownership;
- changing attention/MoE/NCCL behavior;
- replacing graph metadata copy/reference rules;
- adding raw metadata graph-prep;
- making prefix cache default for all runs;
- broad scheduler/cache rewrites.

## Cleanup Tasks

1. Define a clear promoted variant name.

   Suggested name:

   ```text
   dsv4_sm80_a100_victory_prefix_routeb_lifetime
   ```

   It should be the recommended prefix-cache Route B runtime for TARGET 08.30.
   The old diagnostic names such as
   `dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime` may remain
   as aliases or historical variants, but new docs should point to the promoted
   name.

2. Keep verifier support explicit.

   `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1` should remain
   preservable across variant env reset in both perf matrix and text smoke.

3. Document the promoted path.

   Update `prompts/target.md` and `prompts/TARGET_08_radix_prefix_dsv4.md` only
   as needed so new threads know:

   - 08.28 promoted the lifetime cache;
   - 08.29 cleaned up the preset;
   - 08.30 should use the promoted prefix Route B lifetime preset for global
     reprofile.

4. Test the preset.

   At minimum, run:

   ```bash
   python -m py_compile \
     benchmark/offline/deepseek_v4_perf_matrix.py \
     benchmark/offline/deepseek_v4_text_smoke.py \
     python/minisgl/attention/deepseek_v4.py \
     python/minisgl/kernel/deepseek_v4.py

   pytest -q \
     tests/benchmark/test_deepseek_v4_perf_matrix.py \
     tests/benchmark/test_deepseek_v4_text_smoke.py \
     tests/attention/test_deepseek_v4_backend_metadata.py
   ```

   If runtime allows, run a tiny TP8 text smoke with verifier enabled for the
   promoted variant.

## Deliverables

Create:

```text
performance_milestones/target08_route_b_lifetime_promotion_cleanup/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact files changed;
- promoted variant name and env/CLI composition;
- verifier-preservation behavior;
- tests and smoke commands run;
- any aliases kept for historical compatibility;
- final decision: ready for TARGET 08.30, keep cleanup experimental, or split a
  fix.

## Success Criteria

This target is complete if:

- there is one clear promoted Route B lifetime prefix preset for 08.30;
- verifier preservation is tested;
- existing diagnostic variants still work or are deliberately aliased;
- unit tests pass;
- no new runtime mechanism is introduced.

## Stop Rules

Stop and report instead of expanding scope if:

- making a clean preset requires nontrivial runtime refactoring;
- verifier preservation conflicts with variant env isolation;
- graph/correctness failures appear in the smoke sanity check;
- the work drifts into SWA ownership, raw graph prep, attention/MoE/NCCL, or
  low precision.

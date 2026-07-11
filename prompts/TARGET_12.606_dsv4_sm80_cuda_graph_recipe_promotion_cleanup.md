# TARGET 12.606: DSV4 SM80 CUDA Graph Recipe Promotion And Cleanup

## Status

Planned after TARGET 12.605 selects the recipe frontier. Review and expand the
exact soak matrix from its report before execution.

## Purpose

Promote the selected balanced no-env default and named high-concurrency or
long-context presets without environment-variable recipes or manual bucket
lists. Preserve explicit user overrides, fail-closed eager fallback, unified
bucket/reserve telemetry, and the existing numerical/correctness contracts.

Required work will include default/preset wiring, stale opt-in and diagnostic
cleanup, focused unit/text/prefix/long-context/serving soak, repeat-stable
performance and capacity confirmation, and release documentation.

Do not promote a 384/512 policy that TARGET 12.605 classified as research-only
or no-go.

## Output

```text
performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md
```
